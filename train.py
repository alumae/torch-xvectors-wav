"""
Runs a model on a single node across N-gpus.
"""
import os
import sys
from argparse import ArgumentParser
import multiprocessing as mp

import logging
import numpy as np
import torch

from models import XVectorModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

#SEED = 2334
#torch.manual_seed(SEED)
#np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    if hparams.load_model_weights:
        model = XVectorModel.load_from_checkpoint(hparams.load_model_weights)
        
    else:
        model = XVectorModel(hparams)

        

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        weights_summary='top',
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        early_stop_callback=early_stop_callback,
        max_nb_epochs=hparams.max_num_epochs,
        #train_percent_check=0.002,
    )
    

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    mp.set_start_method('fork')

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    
    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed-backend',
        type=str,
        default='ddp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--max-num-epochs', 
        default=200, 
        type=int, 
        metavar='N',
        help='max number of total epochs to run')

    parent_parser.add_argument(
        '--load-model-weights',
        type=str,
        default=None,
        help='Model log directory to restore'
    )

    # each LightningModule defines arguments relevant to it
    parser = XVectorModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
