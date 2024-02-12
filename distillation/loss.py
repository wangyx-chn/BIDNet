import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_calc(outputs,targets,loss_funcs,weights):
    assert len(outputs)==len(targets)==len(loss_funcs)==len(weights),\
    f'The length of outputs, targets, loss_functions, and weights need to be the same, while gets {len(outputs)} {len(targets)} {len(loss_funcs)} {len(weights)}'
    losses = []
    total_loss = 0.0
    for i,output in enumerate(outputs):
        sub_loss = loss_funcs[i](output,targets[i])
        total_loss += sub_loss*weights[i]
        losses.append(sub_loss)
    return total_loss,losses

class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds_S, preds_T):
        
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        sigmoid_pred_T = F.sigmoid(preds_T.permute(0,2,3,1).reshape(-1,C))
        loss = (torch.sum( - sigmoid_pred_T * F.logsigmoid(preds_S.permute(0,2,3,1).reshape(-1,C))))/W/H
        return loss

class SimMat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,RGB_feat,SAR_feat,edge_mask):
        # feat: NCWH
        # bdy: N1WH
        edge_sp = 512
        feat_sp = 256
        N,C,H,W = tuple(SAR_feat.shape)
        SAR_feat = SAR_feat.permute(0,2,3,1).reshape(-1,C)
        RGB_feat = RGB_feat.permute(0,2,3,1).reshape(-1,C)
        valid_mask = (torch.sum(SAR_feat.detach(),dim=1)!=0).cpu()
        
        feat_weight = torch.ones(SAR_feat.shape[0])*valid_mask
        feat_index = torch.multinomial(feat_weight,feat_sp,replacement=False)

        edge_mask = F.interpolate(edge_mask.float().cpu(),[H,W],mode='bilinear',align_corners=True)
        edge_mask = (edge_mask>0.5)*1
        edge_mask = edge_mask.permute(0,2,3,1).reshape(-1)*valid_mask
        if edge_mask.sum()>edge_sp:
            
            edge_weight = feat_weight*edge_mask
            edge_index = torch.multinomial(edge_weight,edge_sp,replacement=False)
        elif edge_mask.sum() == 0:
            
            weight = feat_weight
            edge_index = torch.multinomial(weight,edge_sp,replacement=False)
        else:
            
            edge_weight = feat_weight*edge_mask
            edge_index = torch.multinomial(edge_weight,edge_mask.sum().item(),replacement=False)

        SAR_anch = SAR_feat[edge_index]
        RGB_anch = RGB_feat[edge_index]
        
        SAR_tar = SAR_feat[feat_index]
        RGB_tar = RGB_feat[feat_index]
        SAR_mat = self.cos_sim(SAR_anch,SAR_tar)
        RGB_mat = self.cos_sim(RGB_anch,RGB_tar)
        return RGB_mat,SAR_mat
    
    def cos_sim(self,anch,tar):
        s = torch.cosine_similarity(anch.unsqueeze(1), tar.unsqueeze(0), dim=-1)
        return s


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class ContrastiveLoss(nn.Module):
    def __init__(self, sample_num=256, weight=None, size_average=True):
        super(ContrastiveLoss, self).__init__()
        self.sample_num = sample_num

    def inclass_sample(self,weights,sample_num):
        weights = weights.squeeze()
        try:
            index = torch.multinomial(weights,sample_num)
        except:
            try:
                index = torch.multinomial(weights,weights.size()[0])
            except:
                index = torch.zeros([0,],dtype=torch.long)
        return index.squeeze()

    def featsample(self,SAR_feats,SAR_logits,mask):
        SAR_weight = (mask - SAR_logits).abs()
        weights = torch.pow(SAR_weight,3)
        pos_weights = weights*mask
        neg_weights = weights*(1-mask)
        neg_index = self.inclass_sample(neg_weights,self.sample_num)
        pos_index = self.inclass_sample(pos_weights,self.sample_num)
        
        return SAR_feats[pos_index],SAR_feats[neg_index]

    def forward(self, inputs,logits, protos,mask):
        '''
        inputs:tensor NxC
        logits:tensor Nx1
        mask:Nx1
        protos:list [pos,neg]
        '''
        pos_feats,neg_feats = self.featsample(inputs,logits,mask)
        feats = torch.cat((pos_feats,neg_feats),dim=0)
        pos_sim = torch.cosine_similarity(feats,protos[0].T)
        neg_sim = torch.cosine_similarity(feats,protos[1].T)
        sim = torch.cat((pos_sim.unsqueeze(1),neg_sim.unsqueeze(1)),dim=1)
        label = torch.cat((torch.zeros(pos_feats.size()[0]),torch.ones(neg_feats.size()[0])),dim=0)
        loss_fc = nn.CrossEntropyLoss()
        loss = loss_fc(sim,label.long().cuda())
        return loss
