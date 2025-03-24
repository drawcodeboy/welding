import yaml
from datasets import load_dataset
from models import load_model
from utils import *

import torch
from torch import nn
from torch import optim

import time, argparse

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--config', type=str, default='welding.dnn')
    
    return parser

def main(cfg):
    print(f"=================[{cfg['expr']}]=================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else:
        device = 'cpu'
    print(f"Device: {device}")
    
    # Hyperparameter Settings
    hp_cfg = cfg['hyperparams']
    
    # Load Dataset
    data_cfg = cfg['data']
    train_ds = load_dataset(**data_cfg)
    train_dl = torch.utils.data.DataLoader(train_ds, 
                                           shuffle=True,
                                           batch_size=hp_cfg['batch_size'])
    print(f"Load Dataset {data_cfg['dataset']}")
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(**model_cfg).to(device)
    print(f"Load Model {model_cfg['name']}")
    
    # Load Loss function
    loss_fn = None
    if cfg['loss_fn'] == 'mse_loss':
        loss_fn = nn.MSELoss().to(device)
    else:
        raise Exception('Check loss function')
    
    print(f"Load Loss function {cfg['loss_fn']}")
    
    # Load Optimizer
    optimizer = None
    if hp_cfg['optim'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=hp_cfg['lr'],
                                weight_decay=hp_cfg['weight_decay'])
    elif hp_cfg['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=hp_cfg['lr'],
                               weight_decay=hp_cfg['weight_decay'])
    else:
        raise AssertionError(f"We don\'t support optimizer {hp_cfg['optim']}")
    
    # Load Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=5,
                                                     min_lr=1e-6)
    
    # Train
    total_train_loss = []
    min_loss = 1e4
    
    total_start_time = int(time.time())
    
    for current_epoch in range(1, hp_cfg['epochs']+1):
        print("======================================================")
        print(f"Epoch: [{current_epoch:03d}/{hp_cfg['epochs']:03d}]\n")
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, scheduler, device)
        elapsed_time = int(time.time()) - start_time
        print(f"Train Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s\n")
        
        if train_loss < min_loss:
            min_loss = train_loss
            save_model_ckpt(model_cfg['name'], data_cfg['dataset'], current_epoch,
                            model, cfg['save_path'])

        total_train_loss.append(train_loss)
        save_loss_ckpt(model_cfg['name'], data_cfg['dataset'], total_train_loss, cfg['save_path'])
    
    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()
    
    with open(f'configs/train.{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)