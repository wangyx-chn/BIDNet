import yaml
import argparse
import os
import sys
import shutil
import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorboardX

from eval import eval_net
from loss import loss_calc
import model
from dataset import SN6_Dataset_OneModality as SN6_OM
from dataset import SN6_Dataset_OneModality_wEdge as SN6_OMwE
from torch.utils.data import DataLoader, random_split


def train_net(net,
              config,
              device):
    dir_data = config['dir_data']
    dir_img = config['dir_img']
    dir_gt = config['dir_gt']
    train_file = config['train_file']
    test_file = config['test_file']
    with_edge_flag = config['model']['with_edge']
    if with_edge_flag:
        if config['model']['model_cfg']['with_grad']:
            dir_grad = config['dir_edge']
        else:
            dir_grad = None
        dir_edgegt = config['dir_edgegt']
        train_data = SN6_OMwE(dir_data, dir_img, dir_gt, dir_grad, dir_edgegt, train_file)
        val_data = SN6_OMwE(dir_data, dir_img, dir_gt, dir_grad, dir_edgegt, test_file)
    else:
        train_data = SN6_OM(dir_data, dir_img, dir_gt, train_file)
        val_data = SN6_OM(dir_data, dir_img, dir_gt, test_file)
    n_train = len(train_data)
    print(n_train)
    n_val = len(val_data)
    print(n_val)
    train_loader = DataLoader(train_data, batch_size=config['batchsize'], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_plc = config['lr_policy']
    scheduler = getattr(optim.lr_scheduler,lr_plc['name'])
    lr_plc.pop('name')
    scheduler = scheduler(optimizer, **lr_plc)

    timestamp = time.strftime('%m%d-%H%M%S',time.localtime(time.time()))
    dir_checkpoint = '../work_dirs/{}_{}/'.format(config['cfg_name'],timestamp)
    os.mkdir(dir_checkpoint)
    with open(osp.join(dir_checkpoint,'config.yaml'),'w',encoding='utf-8') as f:
        yaml.dump(config,f)
    writer = tensorboardX.SummaryWriter(osp.join(dir_checkpoint,'runs'))
    
    epochs = config['max_epoch']
    top_acc = 0
    iter=1
    for epoch in range(epochs):
        print("lr=",scheduler.get_last_lr())
        net.train()
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['img']
                true_masks = batch['gt']
                H,W = imgs.shape[-2:]
                H1 = 16 * (H//16)
                W1 = 16 * (W//16)
                imgs = F.interpolate(imgs,[H1,W1])
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                targets = [true_masks]
                loss_seg = getattr(nn,config['loss_functions'][0])
                loss_functions = [loss_seg()]
                if with_edge_flag:
                    
                    edge_gts = batch['edgegt']
                    edge_gts = edge_gts.to(device=device, dtype=torch.float32)
                    
                    edge_supmask = (edge_gts!=0)*1
                    loss_edgeseg = getattr(nn,config['loss_functions'][1])
                    # loss_functions.append(loss_edgeseg(weight=edge_supmask))
                    loss_functions.append(loss_edgeseg())

                    edge_gts = (edge_gts - 1)*edge_supmask
                    targets.append(edge_gts)
                    if config['model']['model_cfg']['with_grad']:
                        edges = batch['edge']
                        edges = F.interpolate(edges,[H1,W1])
                        assert edges.shape[-2:] == imgs.shape[-2:]
                        edges = edges.to(device=device, dtype=torch.float32)
                    else:
                        edges = None
                    outputs = net(imgs,edges)
                else:
                    outputs = net(imgs)
                outputs = list(outputs)
                for i,logits in enumerate(outputs):
                    outputs[i] = F.interpolate(logits,[H,W])    
                
                loss,losses = loss_calc(outputs,targets,loss_functions,config['loss_weights'])

                writer.add_scalar('data/total_loss',loss.item(),iter)
                writer.add_scalar('data/seg_loss',losses[0].item(),iter)
                if with_edge_flag:
                    writer.add_scalar('data/edge_loss',losses[1].item(),iter)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                iter+=1
                pbar.update(imgs.shape[0])
        scheduler.step()
        
        acc_file = open(dir_checkpoint+'acc.txt','a')
        pre,rec,f1,iou,dice,test_loss = eval_net(net,test_loader,device,img_key='img',with_edge=with_edge_flag,edge_key='edge')
        
        writer.add_scalar('loss/test_loss',test_loss,iter)
        
        print("Saving the ckps ...")
        torch.save(net.state_dict(),dir_checkpoint + f'{epoch}.pth')
        acc_file.writelines("epoch{}:pre:{:4f} rec:{:4f} f1:{:4f} IoU:{:4f} dice:{:4f} || \n".format(epoch,pre,rec,f1,iou,dice))
        acc_file.close()



def get_config():
    parser = argparse.ArgumentParser(description='Train a segmentation model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='./config/base.yaml',
                        help='modality of input data')
    args = parser.parse_args()
    config_file = args.config
    filename = osp.splitext(osp.basename(config_file))[0]
    with open(config_file,'r') as f:
        config = yaml.load(f,yaml.Loader)
    if config['cfg_name'] != filename:
        print('cfg_name will be overwritten, please revise it in original .yaml file')
        config['cfg_name'] = filename
    return config



if __name__ == '__main__':
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfgm = config['model']
    net = getattr(model,cfgm['name'])
    net = net(**cfgm['model_cfg'])
    
    if cfgm['pre_model']:
        net.load_state_dict(
            torch.load(cfgm['pre_model'], map_location=device)
        )
    net = nn.DataParallel(net)
    net.to(device=device)
    try:
        train_net(net,
                config,
                device=device
                )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
