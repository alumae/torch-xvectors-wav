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

from data import DiskWavDataset

#SEED = 2334
#torch.manual_seed(SEED)
#np.random.seed(SEED)

def extract_xvectors(trainer, model, extract_xvectors_datadir, store_xvectors_dir, utt2class):
    #model = trainer.get_model()
    breakpoint()
    model.zero_grad()
    model.eval()
    torch.set_grad_enabled(False)
    dataset = DiskWavDataset(datadir=extract_xvectors_datadir, label2id=None, label_file=utt2class)
    dist_sampler = None
    if trainer.use_ddp:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=dist_sampler,
        batch_size=1, #self.hparams.batch_size, 
        collate_fn=dataset.collater,
        num_workers=2)

    for batch_idx, batch in enumerate(dataloader):

        if trainer.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(trainer.data_parallel_device_ids, list):
                root_gpu = trainer.data_parallel_device_ids[0]
            batch = trainer.transfer_batch_to_gpu(batch, root_gpu)
        x = batch["wavs"]
        output = model.extract_xvectors(model.wav_to_features(x))



def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

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
        logger=(hparams.extract_xvectors_datadir is None),
        weights_summary=None,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        early_stop_callback=early_stop_callback,
        max_epochs=hparams.max_num_epochs,
        resume_from_checkpoint=hparams.resume_checkpoint,
        gradient_clip_val=hparams.gradient_clip_val
        #train_percent_check=0.002,
    )
    

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    if (hparams.extract_xvectors_datadir is None):
        trainer.fit(model)
    else:
        trainer.test(model)
        #extract_xvectors(trainer, model, hparams.extract_xvectors_datadir, hparams.store_xvectors_dir, utt2class=hparams.utt2class)



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
        default=-1,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed-backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--max-num-epochs', 
        default=100, 
        type=int, 
        metavar='N',
        help='max number of total epochs to run')

    parent_parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Resume from the specified checkpoint file'
    )

    parent_parser.add_argument(
        '--gradient-clip-val',
        type=float,
        default=0,
        help='Use gradient clipping'
    )



    # each LightningModule defines arguments relevant to it
    parser = XVectorModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
