"""
Example template for defining a system
"""
import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import torchaudio
import torchaudio.transforms

import torchvision.models

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from data import RandomWavChunkSubsetDatasetFactory, WavSegmentDataset, DiskWavDataset
import transforms

from kaldiio import WriteHelper

EPSILON = torch.tensor(torch.finfo(torch.float).eps)

def max_outlier_loss(labels, logits, alpha):
    #breakpoint()
    losses = F.nll_loss(logits, labels, reduction='none')
    outlier_mask = losses >= alpha * losses.max()
    outlier_idx = (outlier_mask == 0).nonzero().squeeze(1)

    loss = losses[outlier_idx].mean()

    return loss


def lsoft_loss(labels, logits, beta):

    labels_one_hot = F.one_hot(labels, logits.size(-1))

    output = torch.exp(logits)
    output = torch.clamp(output, EPSILON, 1)

    labels_update = beta * labels_one_hot + (1 - beta) * output

    losses = -torch.sum(labels_update * logits, dim=-1)

    loss = losses.mean()

    return loss


def nll_loss_with_label_smoothing(inputs, target, smooth_eps=None):
    assert inputs.shape[0] == target.shape[0]
    smooth_eps = smooth_eps or 0

    if smooth_eps < EPSILON:
        return F.nll_loss(inputs, target)

    num_classes = inputs.size(-1)

    eps_sum = smooth_eps / num_classes
    eps_nll = 1. - eps_sum - smooth_eps
    
    likelihood = inputs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    loss = -(eps_nll * likelihood + eps_sum * inputs.sum(-1)).mean(0)

    return loss


# http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf
def reverse_nll_loss(inputs, target, log0):
    lin_inputs = inputs.exp()
    
    num_classes = inputs.size(-1)
    target_rev_nll = lin_inputs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    result = (log0 * lin_inputs).sum(-1) - log0 * target_rev_nll

    return -(result.mean())



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CMN(nn.Module):

    def forward(self, x):
        return x - torch.mean(x, dim=2).unsqueeze(2)



class StatisticalPooling(nn.Module):

    def forward(self, x):
        # x is 3-D with axis [B, feats, T]
        mu = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        result = torch.cat((mu, std), dim=1).squeeze(-1)
        #breakpoint()
        return result


class SelfAttentionPooling(nn.Module):

    def __init__(self, dim, attention_dim, num_heads):
        super(SelfAttentionPooling, self).__init__()
        self.w1 = nn.Conv1d(dim, attention_dim, kernel_size=1, bias=None)
        self.w2 = nn.Conv1d(attention_dim, num_heads, kernel_size=1, bias=None)
        
        #self.w1.weight.data.fill_(0.1)
        #self.w2.weight.data.fill_(0.1)
        self.penalty = None


    def forward(self, x):        
        # put head dimension in front
        #breakpoint()
        a =  torch.clamp(F.softmax(self.w2(F.relu(self.w1(x))), dim=-1), 1e-7, 1).permute(1, 0, 2).unsqueeze(2)
        mu = (x * a).sum(dim=-1, keepdim=True)
        std = ((((x - mu)**2) * a).sum(dim=-1, keepdim=True) + 1e-7).sqrt()
        result = torch.cat((mu, std), dim=2).permute(1, 0, 2, 3).reshape(x.shape[0], -1)

        # Penalty term when multi-head attention is used.
        
        aa = a.squeeze(2).permute(1, 0, 2)
        eye = torch.eye(aa.shape[1], device=aa.device)
        
        self.penalty = (torch.bmm(aa,aa.permute(0,2,1)) - eye.unsqueeze(0).repeat(aa.shape[0], 1, 1) + 1e-7).norm(dim=(1,2)).mean()

        return result

    def get_last_penalty(self):
        return self.penalty



class XVectorModel(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(XVectorModel, self).__init__()
        self.hparams = hparams

        if sum([self.hparams.reverse_ce_alpha > EPSILON, self.hparams.label_smoothing > EPSILON, 
                self.hparams.lsoft_beta != 1.0, self.hparams.max_outlier_alpha != 1.0]) > 1:
            raise Error("Several of  [--lsoft-beta --max.outlier-alpha  --reverse-ce-alpha --label-smoothing] specified, cannot use more than one")

        self.batch_size = hparams.batch_size

        self.chunk_dataset_factory = RandomWavChunkSubsetDatasetFactory(hparams.datadir, 
            batch_size=hparams.batch_size, 
            min_length=2.0, max_length=4.0,
            label_file=hparams.utt2class)

        self.num_outputs = self.chunk_dataset_factory.num_labels

        self.speed_perturbation = transforms.SpeedPerturbation()

        if not self.hparams.no_data_augment:
            self.transforms = [
                transforms.Reverberate(rir_list_filename=hparams.rir_list, sample_rate=self.hparams.sample_rate),
                transforms.AddNoise(noise_list_filename=hparams.noise_list, sample_rate=self.hparams.sample_rate),
                transforms.WhiteNoise(),
            ]

        self.feature_transforms = [
            transforms.FreqMask(replace_with_zero=True)
        ]

        # For dumping x-vectors
        self.write_helper = None

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand((self.batch_size, self.feat_dim, 300))

        # build model
        self.__build_model()

        if hparams.load_pretrained_model is not None:
            logging.info(f"Loading pretraine model (without the final layer) from {hparams.load_pretrained_model}")
            checkpoint = torch.load(hparams.load_pretrained_model)
            del checkpoint['state_dict']["post_pooling_layers.6.weight"]
            del checkpoint['state_dict']["post_pooling_layers.6.bias"]
            self.load_state_dict(checkpoint['state_dict'], strict=False)


    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        bn_momentum = 0.05

        feature_extractor_layers = []
        feature_extractor_layers.append(torchaudio.transforms.MelSpectrogram(
            sample_rate=self.hparams.sample_rate,
            win_length=int(self.hparams.sample_rate * 0.025),
            hop_length=int(self.hparams.sample_rate * 0.01),
            f_min=20.0,
            n_mels=self.hparams.num_fbanks))
        
        feature_extractor_layers.append(torchaudio.transforms.AmplitudeToDB('power', 80.0))

        feature_extractor_layers.append(CMN())
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        if (self.hparams.use_resnet):
            self.pre_resnet =  nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(64, momentum=bn_momentum),
                nn.ReLU(inplace=True))

            self.resnet = torchvision.models.ResNet(SEBasicBlock, [int(i.strip()) for i in self.hparams.resnet_layers.split(",")])
            # kernel_size depends on num_fbanks
            self.post_resnet =  nn.Sequential(
                nn.Conv2d(512, 
                    self.hparams.pre_pooling_hidden_dim, 
                    kernel_size=((self.hparams.num_fbanks - 1) // 2 // 2 // 2 + 1, 1),  
                    bias=False),
                nn.BatchNorm2d(self.hparams.pre_pooling_hidden_dim, momentum=bn_momentum),
                nn.ReLU(inplace=True))
            hidden_dim = 512

        else:           
            pre_pooling_layers = []
            conv_kernel_sizes = [int(i.strip()) for i in self.hparams.conv_kernels.split(",")]
            conv_kernel_dilations = [int(i.strip()) for i in self.hparams.conv_dilations.split(",")]
            assert len(conv_kernel_sizes) == len(conv_kernel_dilations)
            current_input_dim = self.hparams.num_fbanks
            for i in range(len(conv_kernel_sizes)):
                if i < len(conv_kernel_sizes) - 1:
                    hidden_dim = self.hparams.hidden_dim
                else:
                    hidden_dim = self.hparams.pre_pooling_hidden_dim
                conv_layer = nn.Conv1d(current_input_dim, hidden_dim, conv_kernel_sizes[i], 
                                    dilation=conv_kernel_dilations[i], 
                                    padding=(conv_kernel_sizes[i]-1) // 2 * conv_kernel_dilations[i])
                pre_pooling_layers.append(conv_layer)            
                pre_pooling_layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
                pre_pooling_layers.append(nn.ReLU(inplace=True))
                current_input_dim = hidden_dim
            self.pre_pooling_layers = nn.Sequential(*pre_pooling_layers)


        if self.hparams.use_attention:
            self.attention_pooling_layer = SelfAttentionPooling(dim=hidden_dim, attention_dim=self.hparams.attention_dim, num_heads=self.hparams.num_attention_heads)
            self.pooling_layer = self.attention_pooling_layer
            current_input_dim = hidden_dim * self.hparams.num_attention_heads
        else:
            self.pooling_layer = StatisticalPooling()
        
        post_pooling_layers = []

        post_pooling_layers.append(nn.Linear(current_input_dim * 2, self.hparams.hidden_dim))
        post_pooling_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim, momentum=bn_momentum))
        post_pooling_layers.append(nn.ReLU(inplace=True))

        post_pooling_layers.append(nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
        post_pooling_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim, momentum=bn_momentum))
        post_pooling_layers.append(nn.ReLU(inplace=True))

        linear = nn.Linear(self.hparams.hidden_dim, self.num_outputs)
        linear.bias.data.fill_(0.0)
        linear.weight.data.fill_(0.0)

        post_pooling_layers.append(linear)

        post_pooling_layers.append(nn.LogSoftmax(dim=1))
        
        self.post_pooling_layers = nn.Sequential(*post_pooling_layers)

       
        #print(self.model)
        #from torchsummary import summary
        #summary(self.model, input_size=(self.feat_dim, 300), device="cpu")
    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        pooling_output = self._forward_until_pooling(x)
        return self.post_pooling_layers(pooling_output)

    def _forward_until_pooling(self, x):
        if self.hparams.use_resnet:
            x = x.unsqueeze(1)
            x = self.pre_resnet(x)            
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            x = self.post_resnet(x)
            pre_pooling_output = x.view(x.shape[0], -1, x.shape[-1])
        else:
            pre_pooling_output = self.pre_pooling_layers(x)

        return self.pooling_layer(pre_pooling_output)

    def wav_to_features(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)

    def loss(self, labels, logits, smooth_eps, reverse_ce_alpha, reverse_ce_log0, lsoft_beta, max_outlier_alpha):
        if smooth_eps > EPSILON:
            result = nll_loss_with_label_smoothing(logits, labels, smooth_eps)
        elif reverse_ce_alpha > EPSILON:
            result = F.nll_loss(logits, labels) + reverse_ce_alpha * reverse_nll_loss(logits, labels, reverse_ce_log0)
        elif lsoft_beta < 1.0:
            result = lsoft_loss(labels, logits, lsoft_beta)
        elif max_outlier_alpha < 1.0:
            result = max_outlier_loss(labels, logits, max_outlier_alpha)
        else:
            result = F.nll_loss(logits, labels)
        return result

    def extract_xvectors(self, x):
        pooling_output = self._forward_until_pooling(x)
        return self.post_pooling_layers[0](pooling_output)


    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch
        with torch.no_grad():
            if random.random() < self.hparams.speed_perturbation_probability:
                x = self.speed_perturbation(x)
        
        if not self.hparams.no_data_augment:
            if self.hparams.augmix_lambda > EPSILON:
                x_aug_1 = torch.zeros_like(x)
                x_aug_2 = torch.zeros_like(x)        
                with torch.no_grad():
                    for i in range(len(x)):
                        x_aug_1[i] = transforms.augment_and_mix(self.transforms, x[i])
                        x_aug_2[i] = transforms.augment_and_mix(self.transforms, x[i])
                        
                if torch.isnan(x_aug_1).any():
                    logging.warn("NaN in input detected. Replacing it with 0.")
                    x_aug_1 = x.masked_fill(torch.isnan(x_aug_1), 0)
                if torch.isnan(x_aug_2).any():
                    logging.warn("NaN in input detected. Replacing it with 0.")
                    x_aug_2 = x.masked_fill(torch.isnan(x_aug_2), 0)

            else:
                with torch.no_grad():
                    for i in range(len(x)):
                        if self.hparams.use_simple_augment:
                            x[i] = transforms.random_augment(self.transforms, x[i])
                        else:    
                            x[i] = transforms.augment_and_mix(self.transforms, x[i])

        if torch.isnan(x).any():
            logging.warn("NaN in input detected. Replacing it with 0.")
            x = x.masked_fill(torch.isnan(x), 0)

        x_features = self.wav_to_features(x)
        if self.hparams.spec_augment:
            with torch.no_grad():
                for i in range(len(x)):
                    x_features[i] = transforms.augment_and_mix(self.feature_transforms, x_features[i])
        y_hat = self.forward(x_features)

        # calculate loss
        
        loss_val = self.loss(y, y_hat, 
            smooth_eps=self.hparams.label_smoothing,
            reverse_ce_alpha=self.hparams.reverse_ce_alpha,
            reverse_ce_log0=self.hparams.reverse_ce_log0,
            lsoft_beta=self.hparams.lsoft_beta,
            max_outlier_alpha=self.hparams.max_outlier_alpha)

        if torch.isnan(loss_val).any():
            logging.warn("NaN in loss detected. Setting those loss values to zero")
            breakpoint()
            loss_val = loss_val.masked_fill(torch.isnan(loss_val), 0)
            

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        if self.hparams.augmix_lambda > EPSILON:
            p_x = y_hat.exp()
            p_aug_1 = self.forward(self.wav_to_features(x_aug_1)).exp()
            p_aug_2 = self.forward(self.wav_to_features(x_aug_2)).exp()

            log_p_mixture = torch.clamp((p_x + p_aug_1 + p_aug_2) / 3., 1e-7, 1).log()
            
            aug_mix_loss = self.hparams.augmix_lambda * (F.kl_div(log_p_mixture, p_x, reduction='batchmean') +
                        F.kl_div(log_p_mixture, p_aug_1, reduction='batchmean') +
                        F.kl_div(log_p_mixture, p_aug_2, reduction='batchmean')) / 3.
            if self.trainer.use_dp or self.trainer.use_ddp2:
                aug_mix_loss = aug_mix_loss.unsqueeze(0)

            loss_val += aug_mix_loss

        attention_diversity_penalty = None
        if self.hparams.use_attention and self.hparams.num_attention_heads > 1 and self.hparams.attention_diversity_penalty_lambda > EPSILON:
            attention_diversity_penalty = self.hparams.attention_diversity_penalty_lambda * self.attention_pooling_layer.get_last_penalty()
            if self.trainer.use_dp or self.trainer.use_ddp2:
                attention_diversity_penalty = attention_diversity_penalty.unsqueeze(0)
            loss_val += attention_diversity_penalty

        if self.hparams.no_training:
            loss_val = torch.tensor(0.0, requires_grad=True)            
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        if attention_diversity_penalty is not None:
            tqdm_dict["att_div_pen"] = attention_diversity_penalty
        if self.hparams.augmix_lambda > EPSILON:
           tqdm_dict["aug_mix_loss"] = aug_mix_loss

        # can also return just a scalar instead of a dict (return loss_val)
        #breakpoint()
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x = batch["wavs"]
        y = batch["label"]
        
        y_hat = self.forward(self.wav_to_features(x))

        loss_val = self.loss(y, y_hat,
            smooth_eps=0,
            reverse_ce_alpha=0,
            reverse_ce_log0=0,
            lsoft_beta=1.0,
            max_outlier_alpha=1.0)
        

        # acc

        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        #breakpoint()
        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean, 'lr': self.trainer.optimizers[0].param_groups[0]['lr']}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    def test_step(self, batch, batch_idx):
        x = batch["wavs"]
        if self.hparams.dump_xvectors_dir is not None:
            if self.hparams.augmix_xvectors:
                for i in range(len(x)):
                    if self.hparams.use_simple_augment:
                        x[i] = transforms.random_augment(self.transforms, x[i])
                    else:    
                        x[i] = transforms.augment_and_mix(self.transforms, x[i])

            x_features = self.wav_to_features(x)
            if self.hparams.spec_augment:
                with torch.no_grad():
                    for i in range(len(x)):
                        x_features[i] = transforms.augment_and_mix(self.feature_transforms, x_features[i])

            xvectors = self.extract_xvectors(x_features)

            if self.write_helper is None and self.hparams.dump_xvectors_dir is not None:            
                self.write_helper = WriteHelper(f'ark,scp:{self.hparams.dump_xvectors_dir}/xvector.ark,{self.hparams.dump_xvectors_dir}/xvector.scp')
            if self.write_helper:
                for i in range(len(xvectors)):
                    self.write_helper(batch["key"][i], xvectors[i].cpu().numpy())
        elif self.hparams.dump_posteriors_file is not None:
            y_hat = self.forward(self.wav_to_features(x))
            if self.write_helper is None and self.hparams.dump_posteriors_file is not None:            
                self.write_helper = WriteHelper(f'ark:{self.hparams.dump_posteriors_file}')
            if self.write_helper:
                for i in range(len(y_hat)):
                    self.write_helper(batch["key"][i], y_hat[i].cpu().numpy())

        output = OrderedDict({
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_end(self, outputs):
        return {}


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        if self.hparams.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.5)
        elif self.hparams.optimizer_name == "ngd":
            optimizer = NGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.5)
        else:
            raise NotImplementedError()

        #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]


    
    #@pl.data_loader -- we want it to be called every epoch
    def train_dataloader(self):
        dataset = self.chunk_dataset_factory.get_train_dataset(proportion=0.1, sample_rate=self.hparams.sample_rate)
        dist_sampler = None
        num_workers = 8
        if self.trainer.use_ddp:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
            batch_size=None, 
            num_workers=num_workers)


    @pl.data_loader
    def val_dataloader(self):
        dataset = WavSegmentDataset(datadir=self.hparams.dev_datadir, label2id=self.chunk_dataset_factory.label2id, label_file=self.hparams.utt2class, sample_rate=self.hparams.sample_rate)
        dist_sampler = None
        if self.trainer.use_ddp:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
            batch_size=1, #self.hparams.batch_size, 
            collate_fn=dataset.collater,
            num_workers=0)



    @pl.data_loader
    def test_dataloader(self):
        if self.hparams.test_datadir is not None:
            dataset = DiskWavDataset(datadir=self.hparams.test_datadir, label2id=None, label_file=self.hparams.utt2class, sample_rate=self.hparams.sample_rate)
            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
                batch_size=1, #self.hparams.batch_size, 
                collate_fn=dataset.collater,
                num_workers=8)
            return dataloader
        else:
            return None


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        parser.add_argument('--num-fbanks', default=30, type=int)
        #parser.add_argument('--conv-kernels', default="5,1,3,1,3,1,3,1,1")
        #parser.add_argument('--conv-dilations', default="1,1,2,1,3,1,4,1,1")
        parser.add_argument('--conv-kernels', default="5,3,3,1,1")
        parser.add_argument('--conv-dilations', default="1,2,3,1,1")
        parser.add_argument('--hidden-dim', default=512, type=int)
        parser.add_argument('--pre-pooling-hidden-dim', default=1500, type=int)
        parser.add_argument('--learning-rate', default=0.0005, type=float)

        # use attention instead of stats pooling?
        parser.add_argument('--use-attention', default=False,  action='store_true')
        parser.add_argument('--num-attention-heads', default=2, type=int)
        parser.add_argument('--attention-dim', default=128, type=int)
        parser.add_argument('--attention-diversity-penalty-lambda', default=0.01, type=float)

        parser.add_argument('--augmix-lambda', default=0.0, type=float)

        parser.add_argument('--use-resnet', default=False,  action='store_true')
        parser.add_argument('--resnet-layers', default="2,2,2,2", type=str)

        parser.add_argument('--speed-perturbation-probability', default=0.5, type=float)

        parser.add_argument('--use-simple-augment', default=False,  action='store_true')

        # data
        parser.add_argument('--datadir', required=True, type=str)       
        parser.add_argument('--dev-datadir', required=True, type=str)
        parser.add_argument('--test-datadir', required=False, type=str)        
        parser.add_argument('--utt2class', default="utt2lang", type=str)

        # training params (opt)
        parser.add_argument('--optimizer-name', default='adamw', type=str)
        parser.add_argument('--batch-size', default=256, type=int)

        parser.add_argument('--label-smoothing', default=0.0, type=float)

        parser.add_argument('--reverse-ce-alpha', default=0.0, type=float)
        parser.add_argument('--reverse-ce-log0', default=-6.0, type=float)

        parser.add_argument('--lsoft-beta', default=1.0, type=float)

        parser.add_argument('--max-outlier-alpha', default=1.0, type=float)

        parser.add_argument('--rir-list', default="local/rir-list.txt", type=str)
        parser.add_argument('--noise-list', default="local/noise-list.txt", type=str)

        parser.add_argument('--spec-augment', default=False,  action='store_true')

        parser.add_argument('--no-data-augment', default=False,  action='store_true')

        parser.add_argument('--no-training', default=False,  action='store_true')        

        parser.add_argument('--sample-rate', default=16000,  type=int)

        parser.add_argument('--load-pretrained-model', default=None,  type=str)

        parser.add_argument(
            '--extract-xvectors-datadir',
            type=str,
            default=None,
            help='Extract x-vectors for the data in the datadir'
        )
        parser.add_argument(
            '--augmix-xvectors',
            default=False,  action='store_true',
            help='Apply augmix to input data when extracting x-vectors'
        )

        parser.add_argument(
            '--dump-xvectors-dir',
            type=str,
            default=None,
            help='Directory to store xvectors in'
        )

        parser.add_argument(
            '--dump-posteriors-file',
            type=str,
            default=None,
            help='File for storing posteriors of test data'
        )



        return parser
