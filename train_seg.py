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
import random
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.dataset import SegmentationDataset
from model.MeshMamba3D_seg import MeshMamba3D_Seg
import sys
from tqdm import tqdm
import torch.nn.functional as F

sys.setrecursionlimit(3000)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def train(net, optim, criterion, train_dataset, epoch):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    miou=0
    progress = tqdm(enumerate(train_dataset), total=len(train_dataset), 
                    desc=f"Epoch {epoch} Training", leave=False)
    for it, data in progress:
        feats = data["feats"].to(torch.float32).cuda()
        labels = data["label"].cuda()
        adj=data["adj"].cuda()
        batch_size = data["faces"].size(0)
        
        optim.zero_grad()
        n_samples += batch_size


        outputs= net(feats, adj)
        # outputs = outputs.permute(0, 2, 1)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        loss.backward()
        optim.step()
        running_loss += loss.item() * batch_size
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / (n_samples*1024)
    print('epoch: {:} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    message = 'epoch: {:} Train Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)




def test(net, criterion, test_dataset, epoch):
    net.eval()
    acc = 0
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    progress = tqdm(enumerate(test_dataset), total=len(test_dataset),
                   desc=f"Epoch {epoch} Testing", leave=False)
    for i, data in progress:
        feats = data["feats"].to(torch.float32).cuda()
        labels = data["label"].cuda()
        adj=data["adj"].cuda()
        batch_size = data["faces"].size(0)
        n_samples += batch_size
        with torch.no_grad():
            outputs= net(feats, adj)
        # outputs = outputs.permute(0, 2, 1)
            
        
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * batch_size

    epoch_acc = running_corrects.double() / (n_samples*1024)
    epoch_loss = running_loss / n_samples

    if test.best_acc < epoch_acc:
        test.best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join('checkpoints', name, 'best.pkl'))
    print('epoch: {:} test Loss: {:.4f} Acc: {:.4f} Best: {:.4f}'.format(epoch, epoch_loss, epoch_acc,test.best_acc))
    message = 'epoch: {:} test Loss: {:.4f} Acc: {:.4f} Best: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc,
                                                                               test.best_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)


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
    seed_torch(seed=42)
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
    name = config['name']
    
    # ========== Dataset ==========
    augments = []
    if config["augment_scale"]:
        augments.append('scale')
    if config["augment_orient"]:
        augments.append('orient')
    if config["augment_deformation"]:
        augments.append('deformation')
    train_dataset = SegmentationDataset(dataroot, train=True, augments=augments)
    test_dataset = SegmentationDataset(dataroot, train=False)
    print(len(train_dataset))
    print(len(test_dataset))

    train_data_loader = data.DataLoader(train_dataset, num_workers=train_cfg["n_worker"], batch_size=train_cfg["batch_size"],
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=train_cfg["n_worker"], batch_size=train_cfg["batch_size"],
                                       shuffle=False, pin_memory=True)
    
    # ========== Network ==========
    net = MeshMamba3D_Seg(trans_dim=model_cfg["trans_dim"], depth=model_cfg["depth"], drop_path_rate=model_cfg["drop_path_rate"],
                          cls_dim=model_cfg["cls_dim"], num_heads=model_cfg["num_heads"], group_size=model_cfg["group_size"],
                          num_group=model_cfg["num_group"], encoder_dims=model_cfg["encoder_dims"], ordering=model_cfg["ordering"],
                          center_local_k=model_cfg["center_local_k"], bimamba_type=model_cfg["bimamba_type"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # ========== Optimizer ==========
    if train_cfg["optim"].lower() == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    if train_cfg["lr_milestones"].lower() != 'none':
        scheduler = MultiStepLR(optim, milestones=train_cfg["lr_milestones"], gamma=train_cfg["gamma"])
    else:
        scheduler = CosineAnnealingLR(optim, T_max=train_cfg["max_epoch"], eta_min=train_cfg["lr_min"], last_epoch=-1)
    
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join('checkpoints', name)
    #checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')

    os.makedirs(checkpoint_path, exist_ok=True)

    if config["checkpoint"].lower() != 'none':
        net.load_state_dict(torch.load(config["checkpoint"]), strict=False)

    train.step = 0
    test.best_acc = 0

    if config["mode"] == 'train':
        print("begin train")
        for epoch in range(train_cfg["n_epoch"]):
            # train_data_loader.dataset.set_epoch()
            print('iteration', epoch)
            train(net, optim, criterion, train_data_loader, epoch)
            print('train finished')
            test(net, criterion, test_data_loader, epoch)
            print('test finished')
            scheduler.step()
            print(optim.param_groups[0]['lr'])


    else:
        test(net, criterion, test_data_loader, 0)