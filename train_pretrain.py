import torch
import os
import sys
from torch.autograd import Variable
import yaml
from typing import Dict, Any
import argparse
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from model.dataset import ShapeNetDataset
# from model.PointMAE import Point_MAE
from model.MeshMamba3D import MeshMamba3D_MAE
from model.reconstruction import save_results
from utils.util import ClassificationMajorityVoting
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from tqdm import tqdm

def train(net, optim, scheduler, names, train_dataset, epoch):
    net.train()
    running_loss = 0
    epoch_loss = 0

    n_samples = 0
    progress = tqdm(enumerate(train_dataset), total=len(train_dataset), 
                    desc=f"Epoch {epoch} Training", leave=False)
    for it, (faces, feats, Fs, labels, mesh_paths, adj) in progress:
        optim.zero_grad()
        feats = feats.to(torch.float32).cuda()
        faces = faces.to(torch.float32).cuda()
        adj=adj.to(torch.int32).cuda()
        n_samples += faces.shape[0]
        loss = net(feats,adj)
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)

        epoch_loss = running_loss / n_samples
        progress.set_postfix({
            'loss': f'{epoch_loss:.4f}'
        })
    scheduler.step()
    if train.best_loss > epoch_loss:
        train.best_loss = epoch_loss
        train.best_epoch = epoch
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join('checkpoints', names, f'loss-{epoch_loss:.4f}-{epoch:.4f}.pkl'))
    print('epoch ({:}): {:} Train Loss: {:.4f}'.format(names, epoch, epoch_loss))


def test(net,  names, test_dataset, epoch):
    #######################################################################
    # if you are going to show the reconstruct shape, please using the following codes
    #######################################################################
    net.eval()
    progress = tqdm(enumerate(train_dataset), total=len(train_dataset), 
                    desc=f"Epoch {epoch} Training", leave=False)
    for it, (faces, feats, Fs, labels, mesh_paths, adj) in progress:
        feats = feats.to(torch.float32).cuda()
        faces = faces.to(torch.float32).cuda()
        adj=adj.to(torch.int32).cuda()
        n_samples += faces.shape[0]

        with torch.no_grad():
            loss = net(feats,adj)
        running_loss += loss.item() * faces.size(0)

        loss = running_loss / n_samples
        print('Loss: {:.4f}'.format(loss))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse the YAML configuration file, returning the configuration in dictionary format."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML configuration parsing error: {str(e)}")
    
    # 验证关键配置是否存在（可根据实际需求增减）
    required_keys = ['dataroot', 'mode', 'model', 'train']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Configuration file is missing required fields: {key}")
    
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path",type=str, help="YAML configuration file path")
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"Configuration loading failed：{e}", file=sys.stderr)
        sys.exit(1)

    
    mode = config["mode"]
    dataroot = config["dataroot"]
    model_cfg = config['model']
    train_cfg = config['train']

    # ========== Dataset ==========
    augments = []
    if config["augment_scale"]:
        augments.append('scale')
    if config["augment_orient"]:
        augments.append('orient')
    if config["augment_deformation"]:
        augments.append('deformation')

    if 'ShapeNet' in dataroot:
        train_dataset = ShapeNetDataset(dataroot, augment=augments)
        train_dataset.feats.append('center')
        train_dataset.feats.append('normal')

    else:
        train_dataset = ClassificationDataset(dataroot, train=True, augment=augments)
        test_dataset = ClassificationDataset(dataroot, train=False)
        print(len(test_dataset))
        test_data_loader = data.DataLoader(test_dataset, num_workers=train_cfg["n_worker"], batch_size=train_cfg["batch_size"],
                                           shuffle=True, pin_memory=True)
    print(len(train_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=train_cfg["n_worker"], batch_size=train_cfg["batch_size"],
                                        shuffle=True, pin_memory=True)

    # ========== Network ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MeshMamba3D_MAE(mask_ratio=model_cfg["mask_ratio"], mask_type=model_cfg["mask_type"], trans_dim=model_cfg["trans_dim"], 
                          depth=model_cfg["depth"], drop_path_rate=model_cfg["drop_path_rate"], num_heads=model_cfg["num_heads"], 
                          group_size=model_cfg["group_size"], num_group=model_cfg["num_group"], encoder_dims=model_cfg["encoder_dims"],
                          loss=model_cfg["loss"], decoder_depth=model_cfg["decoder_depth"], decoder_num_heads=model_cfg["decoder_num_heads"],
                          decoder_dims=model_cfg["decoder_dims"], decoder_with_lfa=model_cfg["decoder_with_lfa"], ordering=model_cfg["ordering"], 
                          center_local_k=model_cfg["center_local_k"], bimamba_type=model_cfg["bimamba_type"],weight=model_cfg["weight"]).to(device)

    # ========== Optimizer ==========
    if train_cfg["optim"].lower() == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    if train_cfg["lr_milestones"].lower() != 'none':
        ms = train_cfg["lr_milestones"]
        ms = ms.split()
        ms = [int(j) for j in ms]
        scheduler = MultiStepLR(optim, milestones=ms, gamma=0.1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(train_cfg["num_warmup_steps"]),
                                                    num_training_steps=train_cfg["max_epoch"] + 1)

    print(scheduler)

    # ========== MISC ==========

    checkpoint_names = []
    checkpoint_path = os.path.join('checkpoints', config["name"])
    os.makedirs(checkpoint_path, exist_ok=True)

    if config["checkpoint"].lower() != 'none':
        net.load_state_dict(torch.load(config["checkpoint"]), strict=True)


    train.best_loss = 999
    train.best_epoch = 0
    # ========== Start Training ==========

    if config["mode"] == 'train':
        for epoch in range(train_cfg["n_epoch"]):
            print('epoch', epoch)
            train(net, optim, scheduler, config["name"], train_data_loader, epoch)
            print('train finished')



    else:
        test(net, config["name"], test_data_loader, 0)