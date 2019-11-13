"""
Example template for defining a system
"""
import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
from data import RandomChunkSubsetDatasetFactory
from ngd import NGD


from egs import Nnet3EgsDataset



class StatisticalPooling(nn.Module):

    def forward(self, x):
        # x is 3-D with axis [B, feats, T]
        mu = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        result = torch.cat((mu, std), dim=1).squeeze(-1)
        #breakpoint()
        return result

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

        self.chunk_dataset_factory = RandomChunkSubsetDatasetFactory(hparams.datadir, 
            batch_size=hparams.batch_size, 
            min_length=300, max_length=300)
        self.feat_dim = self.chunk_dataset_factory.feat_dim
        self.num_outputs = self.chunk_dataset_factory.num_outputs


        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand((self.batch_size, self.feat_dim, 300))

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
        bn_momentum = 0.05
        conv_kernel_sizes = [int(i.strip()) for i in self.hparams.conv_kernels.split(",")]
        conv_kernel_dilations = [int(i.strip()) for i in self.hparams.conv_dilations.split(",")]
        assert len(conv_kernel_sizes) == len(conv_kernel_dilations)
        layers = []
        current_input_dim = self.feat_dim
        for i in range(len(conv_kernel_sizes)):
            if i < len(conv_kernel_sizes) - 1:
                hidden_dim = self.hparams.hidden_dim
            else:
                hidden_dim = self.hparams.pre_pooling_hidden_dim
            conv_layer = nn.Conv1d(current_input_dim, hidden_dim, conv_kernel_sizes[i], dilation=conv_kernel_dilations[i])
            #conv_layer.bias.data.fill_(0.1)
            #conv_layer.weight.data.normal_(0, 0.1)
            layers.append(conv_layer)            
            layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            current_input_dim = hidden_dim

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
        from torchsummary import summary
        summary(self.model, input_size=(self.feat_dim, 300), device="cpu")
    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(x)

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
        
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)
        #breakpoint()

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        x, y = batch
        
        y_hat = self.forward(x)

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

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return [optimizer], [scheduler]


    
    #@pl.data_loader -- we want it to be called every epoch
    def train_dataloader(self):
        dataset = self.chunk_dataset_factory.get_train_dataset(proportion=0.1)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
            batch_size=None, 
            num_workers=4)


    @pl.data_loader
    def val_dataloader(self):
        dataset = self.chunk_dataset_factory.get_valid_dataset(num_batches=3)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
            batch_size=None, 
            num_workers=1)


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
        parser.add_argument('--conv-kernels', default="5,3,3,1,1")
        parser.add_argument('--conv-dilations', default="1,2,3,1,1")
        parser.add_argument('--hidden-dim', default=512, type=int)
        parser.add_argument('--pre-pooling-hidden-dim', default=1500, type=int)
        parser.add_argument('--learning-rate', default=0.003, type=float)

        # data
        parser.add_argument('--datadir', required=True, type=str)
        parser.add_argument('--num-heldout', default=100, type=int)
        

        # training params (opt)
        parser.add_argument('--optimizer-name', default='adamw', type=str)
        parser.add_argument('--batch-size', default=512, type=int)
        return parser
