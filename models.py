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

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from data import RandomWavChunkSubsetDatasetFactory, WavSegmentDataset
import transforms


from egs import Nnet3EgsDataset

EPSILON = torch.tensor(torch.finfo(torch.float).eps)

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
        a = F.softmax(self.w2(F.relu(self.w1(x))), dim=-1).permute(1, 0, 2).unsqueeze(2)
        mu = (x * a).sum(dim=-1, keepdim=True)
        std = ((((x - mu)**2) * a).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
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

        self.batch_size = hparams.batch_size

        self.chunk_dataset_factory = RandomWavChunkSubsetDatasetFactory(hparams.datadir, 
            batch_size=hparams.batch_size, 
            min_length=2.0, max_length=4.0,
            label_file=hparams.utt2class)

        self.num_outputs = self.chunk_dataset_factory.num_labels

        self.speed_perturbation = transforms.SpeedPerturbation()

        self.transforms = [
            transforms.Reverberate(rir_list_filename="local/real_and_sim_rirs.wavs.txt"),
            transforms.AddNoise(noise_list_filename="local/musan.100.wavs.txt"),
            transforms.WhiteNoise(),
        ]


        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand((self.batch_size, self.feat_dim, 300))

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        feature_extractor_layers = []
        feature_extractor_layers.append(torchaudio.transforms.MelSpectrogram(
            sample_rate=self.chunk_dataset_factory.sample_rate,
            win_length=int(self.chunk_dataset_factory.sample_rate * 0.025),
            hop_length=int(self.chunk_dataset_factory.sample_rate * 0.01),
            f_min=20.0,
            n_mels=self.hparams.num_fbanks))
        
        feature_extractor_layers.append(torchaudio.transforms.AmplitudeToDB('power', 80.0))

        feature_extractor_layers.append(CMN())
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        layers = []
        

        bn_momentum = 0.05
        conv_kernel_sizes = [int(i.strip()) for i in self.hparams.conv_kernels.split(",")]
        conv_kernel_dilations = [int(i.strip()) for i in self.hparams.conv_dilations.split(",")]
        assert len(conv_kernel_sizes) == len(conv_kernel_dilations)
        current_input_dim = self.hparams.num_fbanks
        for i in range(len(conv_kernel_sizes)):
            if i < len(conv_kernel_sizes) - 1:
                hidden_dim = self.hparams.hidden_dim
            else:
                hidden_dim = self.hparams.pre_pooling_hidden_dim
            conv_layer = nn.Conv1d(current_input_dim, hidden_dim, conv_kernel_sizes[i], dilation=conv_kernel_dilations[i])
            layers.append(conv_layer)            
            layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            current_input_dim = hidden_dim

        if self.hparams.use_attention:
            self.attention_pooling_layer = SelfAttentionPooling(dim=hidden_dim, attention_dim=self.hparams.attention_dim, num_heads=self.hparams.num_attention_heads)
            layers.append(self.attention_pooling_layer)
            current_input_dim = hidden_dim * self.hparams.num_attention_heads
        elif self.hparams.use_gating:
            pass
        else:
            layers.append(StatisticalPooling())

        layers.append(nn.Linear(current_input_dim * 2, self.hparams.hidden_dim))
        layers.append(nn.BatchNorm1d(self.hparams.hidden_dim, momentum=bn_momentum))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
        layers.append(nn.BatchNorm1d(self.hparams.hidden_dim, momentum=bn_momentum))
        layers.append(nn.ReLU(inplace=True))

        linear = nn.Linear(self.hparams.hidden_dim, self.num_outputs)
        linear.bias.data.fill_(0.0)
        linear.weight.data.fill_(0.0)

        layers.append(linear)

        layers.append(nn.LogSoftmax(dim=1))

        self.model = nn.Sequential(*layers)
        
        print(self.model)
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
        #breakpoint()
        return self.model(x)

    def wav_to_features(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)

    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll



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
                x = self.speed_perturbation(x).to(x.device)

        x_aug_1 = torch.zeros_like(x)
        x_aug_2 = torch.zeros_like(x)        
        with torch.no_grad():
            for i in range(len(x)):
                x_aug_1[i] = transforms.augment_and_mix(self.transforms, x[i])
                x_aug_2[i] = transforms.augment_and_mix(self.transforms, x[i])


        y_hat = self.forward(self.wav_to_features(x))

        # calculate loss
        loss_val = self.loss(y, y_hat)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        p_x = y_hat.exp()
        p_aug_1 = self.forward(self.wav_to_features(x_aug_1)).exp()
        p_aug_2 = self.forward(self.wav_to_features(x_aug_2)).exp()

        log_p_mixture = torch.clamp((p_x + p_aug_1 + p_aug_2) / 3., 1e-7, 1).log()
        
        aug_mix_loss = 12 * (F.kl_div(log_p_mixture, p_x, reduction='batchmean') +
                    F.kl_div(log_p_mixture, p_aug_1, reduction='batchmean') +
                    F.kl_div(log_p_mixture, p_aug_2, reduction='batchmean')) / 3.
        if self.trainer.use_dp or self.trainer.use_ddp2:
            aug_mix_loss = aug_mix_loss.unsqueeze(0)

        loss_val += aug_mix_loss

            

        attention_diversity_penalty = None
        if self.hparams.use_attention and self.hparams.num_attention_heads > 1:
            attention_diversity_penalty = self.hparams.attention_diversity_penalty_lambda * self.attention_pooling_layer.get_last_penalty()
            if self.trainer.use_dp or self.trainer.use_ddp2:
                attention_diversity_penalty = attention_diversity_penalty.unsqueeze(0)
            loss_val += attention_diversity_penalty
        #breakpoint()


        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        if attention_diversity_penalty is not None:
            tqdm_dict["att_div_pen"] = attention_diversity_penalty

        tqdm_dict["aug_mix_loss"] = aug_mix_loss

        # can also return just a scalar instead of a dict (return loss_val)
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

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

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
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]


    
    #@pl.data_loader -- we want it to be called every epoch
    def train_dataloader(self):
        dataset = self.chunk_dataset_factory.get_train_dataset(proportion=0.1)
        dist_sampler = None
        num_workers = 4
        if self.trainer.use_ddp:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
            batch_size=None, 
            num_workers=num_workers)


    @pl.data_loader
    def val_dataloader(self):
        dataset = WavSegmentDataset(datadir=self.hparams.dev_datadir, label2id=self.chunk_dataset_factory.label2id, label_file=self.hparams.utt2class)
        dist_sampler = None
        if self.trainer.use_ddp:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
            batch_size=1, #self.hparams.batch_size, 
            collate_fn=dataset.collater,
            num_workers=0)


    @pl.data_loader
    def test_dataloader(self):
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

        #parser.add_argument('--conv-kernels', default="5,1,3,1,3,1,3,1,1")
        #parser.add_argument('--conv-dilations', default="1,1,2,1,3,1,4,1,1")
        parser.add_argument('--num-fbanks', default=30, type=int)

        parser.add_argument('--conv-kernels', default="5,3,3,1,1")
        parser.add_argument('--conv-dilations', default="1,2,3,1,1")
        parser.add_argument('--hidden-dim', default=512, type=int)
        parser.add_argument('--pre-pooling-hidden-dim', default=1500, type=int)
        parser.add_argument('--learning-rate', default=0.003, type=float)

        # use attention instead of stats pooling?
        parser.add_argument('--use-attention', default=False, type=bool)
        parser.add_argument('--num-attention-heads', default=2, type=int)
        parser.add_argument('--attention-dim', default=128, type=int)
        parser.add_argument('--attention-diversity-penalty-lambda', default=0.01, type=float)

        # use a dedicated gating branch before pooling
        parser.add_argument('--use-attention', default=False, type=bool)

        parser.add_argument('--speed-perturbation-probability', default=0.5, type=float)

        # data
        parser.add_argument('--datadir', required=True, type=str)       
        parser.add_argument('--dev-datadir', required=True, type=str)
        parser.add_argument('--utt2class', default="utt2lang", type=str)

        # training params (opt)
        parser.add_argument('--optimizer-name', default='adamw', type=str)
        parser.add_argument('--batch-size', default=256, type=int)
        return parser
