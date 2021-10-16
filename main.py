import os
import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, wandb

from data import DInterface
from model import MInterface
# from model.cnncbam import CnnCbam
# from utils import load_model_path_by_dict

warnings.filterwarnings("ignore")

args={
    # Basic Training Control
    "project_name" : "Breast_PL",
    "batch_size" : 16,
    "batch_accumulate" : 2,
    "num_workers" : 4,
    "seed" : 42,
    "lr" : 1e-3,

    # Training Info
    "dataset" : "breast_data",
    "data_dir" : r"G:\我的雲端硬碟\Lab\Project\胸大肌\乳腺\BreastCNN\data\histogram_cc",
    "class_num" : 2,
    "model_name" : "cnn_cbam",
    "loss" : "focal",
    "weight_decay" : 1e-5,
    "no_augment" : False,
    "aug_prob" : 0.5,
    "log_dir" : "lightning_logs",
    "scale" : True,
    

    #LR Scheduler
    "lr_scheduler" : 'cosine',
    "lr_decay_steps" : 20,
    "lr_decay_rate" : 0.5,
    "lr_decay_min_lr" : 1e-5,

    # Model Hyperparameters

    # Loss Hyperparameters

    # Restart Control
    "load_best" : True,
    "load_v_num" : 0,
    "load_dir" : r"",
}

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(
    #         logging_interval='epoch'))

    return callbacks


def main(args):
    pl.seed_everything(args["seed"])
    # load_path = load_model_path_by_args(args)
    # data_module = DInterface(**vars(args))
    data_module = DInterface(**args)
    model = MInterface(**args)


    # if load_path is None:
    #     model = MInterface(**vars(args))
    # else:
    #     model = MInterface(**vars(args))
    #     args.resume_from_checkpoint = load_path


    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    wandb = WandbLogger(project=args["project_name"], log_model='all')
    # args.callbacks = load_callbacks()
    # args.logger = logger

    # trainer = Trainer.from_argparse_args(args)

    trainer = Trainer(
        accumulate_grad_batches=args["batch_accumulate"],
        auto_scale_batch_size='binsearch',
        auto_lr_find=False,
        max_epochs=1,       # 1000
        min_epochs=1,       # 200
        log_every_n_steps=args["batch_size"],
        gpus=1,
        precision=32,
        logger=wandb,
        callbacks=load_callbacks()
        )
    wandb.watch(model)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

if __name__ == '__main__':
    main(args)
