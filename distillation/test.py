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
import cv2

from eval import eval_net
from loss import loss_calc
from vis_utils import exist_or_create,cover_or_create
import models_w_feature as models
from dataset import SN6_Dataset_OneModality as SN6_OM
from dataset import SN6_Dataset_OneModality_wEdge as SN6_OMwE
from torch.utils.data import DataLoader, random_split


def test_net(net, config, device):
    dir_data = config['dir_data']
    dir_img = config['dir_img']
    dir_gt = config['dir_gt']
    test_file = config['test_file']
    with_edge_flag = config['model']['model_cfg']['with_grad'] and config['model']['with_edge']
    if with_edge_flag:
        dir_edge = config['dir_edge']
        dir_edgegt = config['dir_edgegt']
        val_data = SN6_OMwE(dir_data, dir_img, dir_gt, dir_edge, dir_edgegt, test_file)
    else:
        val_data = SN6_OM(dir_data, dir_img, dir_gt, test_file)
    test_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    net.eval()
    mask_type = torch.float32
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    n_test = len(test_loader)

    timestamp = time.strftime('%m%d-%H%M%S',time.localtime(time.time()))
    wk_dir = '/home/wyx/SN6_extraction/SN6/work_dirs_results/{}_{}/'.format(config['cfg_name'],timestamp)
    os.mkdir(wk_dir)
    os.mkdir(osp.join(wk_dir,'seg_prob'))
    os.mkdir(osp.join(wk_dir,'seg_map'))
    with open(osp.join(wk_dir,'config.yaml'),'w',encoding='utf-8') as f:
        yaml.dump(config,f)

    if config['ori_features']:
        modality = config['modality']
        backbone = config['model']['model_cfg']['encoder_name']
        if not config['model']['with_edge']:
            folder = 'wo_edge'
        elif not config['model']['model_cfg']['with_grad']:
            folder = 'wo_edge'
        else:
            folder = config['dir_edge'].split('/')[-1]
        
        dir = osp.join(dir_data,f'orifeat_{modality}')
        exist_or_create(dir)
        dir = osp.join(dir,backbone)
        exist_or_create(dir)
        dir = osp.join(dir,folder)
        cover_or_create(dir)
    with tqdm(total=n_test, desc='Validation', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            imgs = batch['img']
            true_masks = batch['gt']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            H,W = imgs.shape[-2:]
            H1 = 16 * (H//16)
            W1 = 16 * (W//16)
            imgs = F.interpolate(imgs,[H1,W1])

            if with_edge_flag:
                edges = batch['edge']
                edges = edges.to(device=device, dtype=torch.float32)
                edges = F.interpolate(edges,[H1,W1])
                with torch.no_grad():
                    output = net(imgs,edges)
                    seg_logits = output['seg_logits']
                    seg_logits = F.interpolate(seg_logits,[H,W])
            else:
                with torch.no_grad():
                    output = net(imgs)
                    seg_logits = output['seg_logits']
                    seg_logits = F.interpolate(seg_logits,[H,W])
            
            filename = batch['file_name']
            seg_prob = torch.sigmoid(seg_logits)
            pred = (seg_prob > config['prob_thr']).float()
            for i,file in enumerate(filename):
                cv2.imwrite(osp.join(wk_dir,'seg_prob',file),seg_prob[i][0].cpu().numpy()*255)
                cv2.imwrite(osp.join(wk_dir,'seg_map',file),pred[i][0].cpu().numpy()*255)
            
            
            if config['ori_features']:
                for feat_name in config['features']:
                    save_ori_feat(dir,feat_name,filename,output[feat_name])
            if config['vis_features']:
                for feat_name in config['features']:
                        save_feat(wk_dir,feat_name,filename,output[feat_name],config['featuremap_mode'])

            if config['evaluate']:
                dt = pred.view(-1)
                gt = true_masks.view(-1)
                tp = torch.sum(dt*gt)
                fp = torch.sum(dt*(1-gt))
                tn = torch.sum((1-dt)*(1-gt))
                fn = torch.sum((1-dt)*gt)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
            pbar.update(imgs.size(0))
    if config['evaluate']:
        iou = TP/(TP+FP+FN)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = 2*precision*recall/(precision+recall)
        dice = 2*TP/(FP+2*TP+FN)
        print("pre:{:4f}\nrec:{:4f}\nf1:{:4f}\nIoU:{:4f}\ndice:{:4f}".format(precision,recall,f1,iou,dice))
        return precision,recall,f1,iou,dice
    else:
        return 1

def save_feat(dir,feat_name,filename,feats,mode='ave'):
    if not isinstance(feats,list):
        feats = [feats]
    for i,feat in enumerate(feats):
        C,H,W = feat.shape[-3:]
        save_dir = osp.join(dir,feat_name+f'_{i}_{C}x{H}x{W}')
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        feat = feat.cpu().numpy()
        for j,file in enumerate(filename):
            if mode=='ave':
                featmap = np.mean(feat[j],axis=0)
            if mode=='max':
                featmap = np.max(feat[j],axis=0)
            minv = np.percentile(featmap,5)
            maxv = np.percentile(featmap,95)
            featmap = ((np.clip(featmap,minv,maxv)-minv)/(maxv-minv)*255).astype(np.uint8)
            #featmap = cv2.applyColorMap(featmap,cv2.COLORMAP_WINTER)
            cv2.imwrite(osp.join(save_dir,file),featmap)

def save_ori_feat(dir,feat_name,filename,feats):
    pre = feat_name.split('_')[0]
    if isinstance(feats,list):
        for i,f in enumerate(feats):
            save_dir = osp.join(dir,f'{pre}_{i}')
            exist_or_create(save_dir)
            torch.save(f[0].cpu(),osp.join(save_dir,filename[0].replace('png','pt')))
    else:
        save_dir = osp.join(dir,pre)
        exist_or_create(save_dir)
        torch.save(feats[0].cpu(),osp.join(save_dir,filename[0].replace('png','pt')))




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
    net = getattr(models,cfgm['name'])
    net = net(**cfgm['model_cfg'])
    
    assert cfgm['model_weight'] , 'the weight file is not given'
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(cfgm['model_weight'], map_location=device))
    net.to(device=device)
    try:
        test_net(net,
                config,
                device=device
                )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)