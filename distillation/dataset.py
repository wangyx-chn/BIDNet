import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random
class SN6_Dataset_OneModality(Dataset):
    def __init__(self, dir_data, dir_img, dir_gt, namelist):
        self.dir_data = dir_data
        self.dir_img = dir_img
        self.dir_gt = dir_gt
        self.namelist = namelist
        #assert split in ["train","val","test"], 'split must be in ["train","val","test"]'
        with open(osp.join(self.dir_data,self.namelist),'r') as filename:
            self.img_names = [name.strip() for name in filename.readlines()]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        gt_file = osp.join(self.dir_data, self.dir_gt, self.img_names[i])
        img_file = osp.join(self.dir_data, self.dir_img, self.img_names[i])

        gt = cv2.imread(gt_file,flags=0)/255
        gt = np.expand_dims(gt,axis=0)
        img = cv2.imread(img_file)/255
        img = img.transpose((2,0,1)) #H,W,C to C,H,W

        return {
            'img':torch.from_numpy(img).type(torch.FloatTensor),
            'gt':torch.from_numpy(gt).type(torch.FloatTensor),
            'file_name':self.img_names[i]
        }

class SN6_Dataset_OneModality_wEdge(Dataset):
    def __init__(self, dir_data, dir_img, dir_gt, dir_edge, dir_edgegt, namelist):
        self.dir_data = dir_data
        self.dir_img = dir_img
        self.dir_gt = dir_gt
        self.dir_edge = dir_edge
        self.dir_edgegt = dir_edgegt
        self.namelist = namelist
        #assert split in ["train","val","test"], 'split must be in ["train","val","test"]'
        with open(osp.join(self.dir_data,self.namelist),'r') as filename:
            self.img_names = [name.strip() for name in filename.readlines()]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        gt_file = osp.join(self.dir_data, self.dir_gt, self.img_names[i])
        img_file = osp.join(self.dir_data, self.dir_img, self.img_names[i])
        
        edgegt_file = osp.join(self.dir_data, self.dir_edgegt, self.img_names[i])
        gt = cv2.imread(gt_file,flags=0)/255
        gt = np.expand_dims(gt,axis=0)
        edgegt = cv2.imread(edgegt_file,flags=0)
        edgegt = np.expand_dims(edgegt,axis=0)
        img = cv2.imread(img_file)/255
        img = img.transpose((2,0,1)) #H,W,C to C,H,W

        data = {'img':torch.from_numpy(img).type(torch.FloatTensor),
                'gt':torch.from_numpy(gt).type(torch.FloatTensor),
                'edge':0,
                'edgegt':torch.from_numpy(edgegt).type(torch.FloatTensor),
                'file_name':self.img_names[i]}
        if self.dir_edge:
            edge_file = osp.join(self.dir_data, self.dir_edge, self.img_names[i])
            edge = cv2.imread(edge_file,flags=0)/255
            edge = np.expand_dims(edge,axis=0)
            data['edge'] = torch.from_numpy(edge).type(torch.FloatTensor)
        return data

class SN6_Dataset_TwoModality(Dataset):
    def __init__(self, dir_data, dir_img, dir_gt, feat_list, feat_dirs, namelist):
        self.dir_data = dir_data
        self.dir_img = dir_img
        self.dir_gt = dir_gt
        self.feat_list = feat_list
        self.feat_dirs = feat_dirs
        self.namelist = namelist
        #assert split in ["train","val","test"], 'split must be in ["train","val","test"]'
        with open(osp.join(self.dir_data,self.namelist),'r') as filename:
            self.img_names = [name.strip() for name in filename.readlines()]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        gt_file = osp.join(self.dir_data, self.dir_gt, self.img_names[i])
        img_file = osp.join(self.dir_data, self.dir_img, self.img_names[i])

        gt = cv2.imread(gt_file,flags=0)/255
        gt = np.expand_dims(gt,axis=0)
        img = cv2.imread(img_file)/255
        img = img.transpose((2,0,1)) #H,W,C to C,H,W

        batch =  {
            'img':torch.from_numpy(img).type(torch.FloatTensor),
            'gt':torch.from_numpy(gt).type(torch.FloatTensor),
            'file_name':self.img_names[i]
        }
        for j,feat in enumerate(self.feat_list):
            featmap = torch.load(osp.join(self.feat_dirs[j],self.img_names[i].replace('png','pt')))
            batch[feat] = featmap
        return batch

class SN6_Dataset_TwoModality_wEdge(Dataset):
    def __init__(self, dir_data, dir_img, dir_gt, dir_edge, dir_edgegt, feat_list, feat_dirs, namelist):
        self.dir_data = dir_data
        self.dir_img = dir_img
        self.dir_gt = dir_gt
        self.dir_edge = dir_edge
        self.dir_edgegt = dir_edgegt
        self.feat_list = feat_list
        self.feat_dirs = feat_dirs
        self.namelist = namelist
        #assert split in ["train","val","test"], 'split must be in ["train","val","test"]'
        with open(osp.join(self.dir_data,self.namelist),'r') as filename:
            self.img_names = [name.strip() for name in filename.readlines()]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        gt_file = osp.join(self.dir_data, self.dir_gt, self.img_names[i])
        img_file = osp.join(self.dir_data, self.dir_img, self.img_names[i])
        
        edgegt_file = osp.join(self.dir_data, self.dir_edgegt, self.img_names[i])
        gt = cv2.imread(gt_file,flags=0)/255
        gt = np.expand_dims(gt,axis=0)
        edgegt = cv2.imread(edgegt_file,flags=0)
        edgegt = np.expand_dims(edgegt,axis=0)
        img = cv2.imread(img_file)/255
        img = img.transpose((2,0,1)) #H,W,C to C,H,W

        batch = {'img':torch.from_numpy(img).type(torch.FloatTensor),
                'gt':torch.from_numpy(gt).type(torch.FloatTensor),
                'edge':0,
                'edgegt':torch.from_numpy(edgegt).type(torch.FloatTensor),
                'file_name':self.img_names[i]}
        if self.dir_edge:
            edge_file = osp.join(self.dir_data, self.dir_edge, self.img_names[i])
            edge = cv2.imread(edge_file,flags=0)/255
            edge = np.expand_dims(edge,axis=0)
            batch['edge'] = torch.from_numpy(edge).type(torch.FloatTensor)
        for j,feat in enumerate(self.feat_list):
            featmap = torch.load(osp.join(self.feat_dirs[j],self.img_names[i].replace('png','pt')))
            batch[feat] = featmap
        return batch

class SN6_Dataset_TwoModalityOnline(Dataset):
    def __init__(self, dir_data, dir_RGBimg, dir_SARimg, dir_gt,  dir_edgegt,namelist,size=900):
        self.dir_data = dir_data
        self.dir_RGBimg = dir_RGBimg
        self.dir_SARimg = dir_SARimg
        self.dir_gt = dir_gt
        self.dir_edgegt = dir_edgegt
        self.namelist = namelist
        self.size = size
        #assert split in ["train","val","test"], 'split must be in ["train","val","test"]'
        with open(osp.join(self.dir_data,self.namelist),'r') as filename:
            self.img_names = [name.strip() for name in filename.readlines()]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        
        gt_file = osp.join(self.dir_data, self.dir_gt, self.img_names[i])
        RGBimg_file = osp.join(self.dir_data, self.dir_RGBimg, self.img_names[i])
        SARimg_file = osp.join(self.dir_data, self.dir_SARimg, self.img_names[i])
        
        gt = cv2.imread(gt_file,flags=0)/255
        gt = np.expand_dims(gt,axis=0)
        size = self.size
        H,W = gt.shape[-2:]
        if H>size and W>size:
            h0 = random.randint(0,H-size)
            w0 = random.randint(0,W-size)
        else:
            h0 = 0
            w0 = 0
        gt = gt[:,h0:h0+size,w0:w0+size]
        edgegt_file = osp.join(self.dir_data, self.dir_edgegt, self.img_names[i])
        edgegt = cv2.imread(edgegt_file,flags=0)
        edgegt = np.expand_dims(edgegt,axis=0)
        edgegt = edgegt[:,h0:h0+size,w0:w0+size]
        RGBimg = cv2.imread(RGBimg_file)/255
        RGBimg = RGBimg.transpose((2,0,1)) #H,W,C to C,H,W
        RGBimg = RGBimg[:,h0:h0+size,w0:w0+size]
        SARimg = cv2.imread(SARimg_file)/255
        SARimg = SARimg.transpose((2,0,1)) #H,W,C to C,H,W
        SARimg = SARimg[:,h0:h0+size,w0:w0+size]

        batch =  {
            'img_T':torch.from_numpy(RGBimg).type(torch.FloatTensor),
            'img':torch.from_numpy(SARimg).type(torch.FloatTensor),
            'gt':torch.from_numpy(gt).type(torch.FloatTensor),
            'edgegt':torch.from_numpy(edgegt).type(torch.FloatTensor),
            'file_name':self.img_names[i]
        }
        return batch

class SN6_Dataset_TwoModalityOnline_wEdge(Dataset):
    def __init__(self, dir_data, dir_RGBimg, dir_SARimg, dir_gt, dir_RGBedge, dir_SARedge, dir_edgegt, namelist, size=900):
        self.dir_data = dir_data
        self.dir_RGBimg = dir_RGBimg
        self.dir_SARimg = dir_SARimg
        self.dir_gt = dir_gt
        self.dir_RGBedge = dir_RGBedge
        self.dir_SARedge = dir_SARedge
        self.dir_edgegt = dir_edgegt
        self.namelist = namelist
        self.size = size
        #assert split in ["train","val","test"], 'split must be in ["train","val","test"]'
        with open(osp.join(self.dir_data,self.namelist),'r') as filename:
            self.img_names = [name.strip() for name in filename.readlines()]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        gt_file = osp.join(self.dir_data, self.dir_gt, self.img_names[i])
        RGBimg_file = osp.join(self.dir_data, self.dir_RGBimg, self.img_names[i])
        SARimg_file = osp.join(self.dir_data, self.dir_SARimg, self.img_names[i])
        
        edgegt_file = osp.join(self.dir_data, self.dir_edgegt, self.img_names[i])
        gt = cv2.imread(gt_file,flags=0)/255
        gt = np.expand_dims(gt,axis=0)

        size = self.size
        H,W = gt.shape[-2:]
        if H>size and W>size:
            h0 = random.randint(0,H-size)
            w0 = random.randint(0,W-size)
        else:
            h0 = 0
            w0 = 0
        gt = gt[:,h0:h0+size,w0:w0+size]
        edgegt = cv2.imread(edgegt_file,flags=0)
        edgegt = np.expand_dims(edgegt,axis=0)
        edgegt = edgegt[:,h0:h0+size,w0:w0+size]
        RGBimg = cv2.imread(RGBimg_file)/255
        RGBimg = RGBimg.transpose((2,0,1)) #H,W,C to C,H,W
        RGBimg = RGBimg[:,h0:h0+size,w0:w0+size]
        SARimg = cv2.imread(SARimg_file)/255
        SARimg = SARimg.transpose((2,0,1)) #H,W,C to C,H,W
        SARimg = SARimg[:,h0:h0+size,w0:w0+size]

        batch = {'img_T':torch.from_numpy(RGBimg).type(torch.FloatTensor),
                'img':torch.from_numpy(SARimg).type(torch.FloatTensor),
                'gt':torch.from_numpy(gt).type(torch.FloatTensor),
                'edge_T':0,
                'edge':0,
                'edgegt':torch.from_numpy(edgegt).type(torch.FloatTensor),
                'file_name':self.img_names[i]}
        if self.dir_SARedge:
            edge_file = osp.join(self.dir_data, self.dir_SARedge, self.img_names[i])
            edge = cv2.imread(edge_file,flags=0)/255
            edge = np.expand_dims(edge,axis=0)
            edge = edge[:,h0:h0+size,w0:w0+size]
            batch['edge'] = torch.from_numpy(edge).type(torch.FloatTensor)
            edge_T_file = osp.join(self.dir_data, self.dir_RGBedge, self.img_names[i])
            edge_T = cv2.imread(edge_T_file,flags=0)/255
            edge_T = np.expand_dims(edge_T,axis=0)
            edge_T = edge_T[:,h0:h0+size,w0:w0+size]
            batch['edge_T'] = torch.from_numpy(edge_T).type(torch.FloatTensor)
        return batch