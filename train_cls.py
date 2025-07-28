# baseline, main
import torch
import os
import sys
from torch.autograd import Variable
import argparse
import yaml
from typing import Dict, Any
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from model.dataset import ClassificationDataset
from model.MeshMamba3D import MeshMamba3D
from utils.util import ClassificationMajorityVoting
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from tqdm import tqdm
from model.MeshMamba3D import MeshMamba3D

def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_embedding(data, label, title):

    print(len(label))
    cmap = cm.rainbow(np.linspace(0, 1, len(label)))

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()

    for i in range(data.shape[0]):
        c = cm.rainbow(int(255 / 40 * label[i]))
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 40),  fontdict={'weight': 'bold', 'size': 7})
        plt.scatter(data[i, 0], data[i, 1], color=c, alpha=0.5)
    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=9)

    return fig


def train(net, optim, scheduler, names, criterion, train_dataset, epoch, model_cfg):
    net.train()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    # 使用tqdm包装数据加载器
    progress = tqdm(enumerate(train_dataset), total=len(train_dataset), 
                    desc=f"Epoch {epoch} Training", leave=False)
    for it, data in progress:
        optim.zero_grad()
        feats = data["feats"].to(torch.float32).cuda()
        labels = data["label"].cuda()
        adj=data["adj"].cuda()
        batch_size = data["faces"].size(0)
        n_samples += batch_size
        
        outputs = net(feats, adj)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optim.step()

        # 更新统计量
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        
        # 更新进度条信息
        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.float() / n_samples
        progress.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.4f}'
        })

    scheduler.step()
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects.float() / n_samples
    tqdm.write(f'Epoch ({names}): {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    message = f'epoch ({names}): {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n'
    with open(os.path.join('checkpoints', names, 'log.txt'), 'a') as f:
        f.write(message)

def test(net, names, criterion, test_dataset, epoch, model_cfg):
    net.eval()
    voted = ClassificationMajorityVoting(model_cfg["cls_dim"])
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    # 使用tqdm包装测试数据加载器
    progress = tqdm(enumerate(test_dataset), total=len(test_dataset),
                   desc=f"Epoch {epoch} Testing", leave=False)
    for i, data in progress:
        feats = data["feats"].to(torch.float32).cuda()
        labels = data["label"].cuda()
        adj=data["adj"].cuda()
        batch_size = data["faces"].size(0)
        n_samples += batch_size

        with torch.no_grad():
            outputs = net(feats,adj)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        voted.vote(data["path"], preds, labels)

        # 更新进度条信息
        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.float() / n_samples
        progress.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.4f}'
        })

    # 计算最终指标
    epoch_acc = running_corrects.float() / n_samples
    epoch_loss = running_loss / n_samples
    epoch_vacc = voted.compute_accuracy()
    
    if test.best_acc < epoch_acc:
        test.best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        #torch.save(best_model_wts, os.path.join('checkpoints', names, f'acc-{epoch_acc:.4f}-{epoch:.4f}.pkl'))
        torch.save(best_model_wts, os.path.join('checkpoints', names, 'best_acc.pkl'))
    if test.best_vacc < epoch_vacc:
        test.best_vacc = epoch_vacc
        best_model_wts = copy.deepcopy(net.state_dict())
        #torch.save(best_model_wts, os.path.join('checkpoints', names, f'vacc-{epoch_vacc:.4f}-{epoch:.4f}.pkl'))
        torch.save(best_model_wts, os.path.join('checkpoints', names, 'best_vacc.pkl'))
    
    # 使用tqdm.write避免打断进度条
    message = (f'Epoch ({names}): {epoch} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
               f'Best Acc: {test.best_acc:.4f}\n'
               f'Test acc [voted] = {epoch_vacc:.4f} Best acc [voted] = {test.best_vacc:.4f}')
    tqdm.write(message)
    with open(os.path.join('checkpoints', names, 'log.txt'), 'a') as f:
        f.write(message + '\n')

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
    seed_torch(seed=43)
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
    train_dataset = ClassificationDataset(dataroot, train=True, augment=augments)
    train_dataset.feats.append('center')
    train_dataset.feats.append('normal')
    test_dataset = ClassificationDataset(dataroot, train=False)
    test_dataset.feats.append('center')
    test_dataset.feats.append('normal')
    print(len(train_dataset))
    print(len(test_dataset))

    train_data_loader = data.DataLoader(train_dataset, num_workers=train_cfg["n_worker"], batch_size=train_cfg["batch_size"],
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=train_cfg["n_worker"], batch_size=train_cfg["batch_size"],
                                       shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ========== Network ==========
    net = MeshMamba3D(trans_dim=model_cfg["trans_dim"], depth=model_cfg["depth"], drop_path_rate=model_cfg["drop_path_rate"], 
                      cls_dim=model_cfg["cls_dim"], num_heads=model_cfg["num_heads"], group_size=model_cfg["group_size"], 
                      num_group=model_cfg["num_group"], encoder_dims=model_cfg["encoder_dims"], ordering=model_cfg["ordering"],
                      center_local_k=model_cfg["center_local_k"], bimamba_type=model_cfg["bimamba_type"]).to(device)

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
                                                    num_training_steps=train_cfg["n_epoch"] )

    print(scheduler)
    criterion = nn.CrossEntropyLoss()
    checkpoint_names = []
    checkpoint_path = os.path.join('checkpoints', config["name"])

    os.makedirs(checkpoint_path, exist_ok=True)

    if config["checkpoint"].lower() != 'none':
        net.load_state_dict(torch.load(config["checkpoint"]), strict=False)

    train.step = 0
    test.best_acc = 0
    test.best_vacc = 0

    # ========== Start Training ==========

    if config["mode"] == 'train':
        for epoch in range(config["n_epoch"]):
            # train_data_loader.dataset.set_epoch(epoch)
            print('epoch', epoch)
            train(net, optim, scheduler, config["name"], criterion, train_data_loader, epoch, model_cfg)
            print('train finished')
            test(net, config["name"], criterion, test_data_loader, epoch, model_cfg)
            print('test finished')



    else:

        test(net, config["name"], criterion, test_data_loader, 0, model_cfg)
