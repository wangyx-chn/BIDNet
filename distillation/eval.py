import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def eval_net(net, loader, device, img_key, gt_key='gt', with_edge=False, edge_key=None):
    assert (not with_edge) or edge_key is not None , 'edge_key is not given'
    #if with_edge, the edge_key is needed
    net.eval()
    mask_type = torch.float32
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    loss = 0.0
    n_test = len(loader)
    
    with tqdm(total=n_test, desc='Validation', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch[img_key]
            true_masks = batch[gt_key]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            H,W = imgs.shape[-2:]
            H1 = 16 * (H//16)
            W1 = 16 * (W//16)
            imgs = F.interpolate(imgs,[H1,W1])

            if with_edge:
                edges = batch[edge_key]
                if not isinstance(edges,int):
                    edges = edges.to(device=device, dtype=torch.float32)
                    edges = F.interpolate(edges,[H1,W1])
                with torch.no_grad():
                    outputs = net(imgs,edges)
                    
            else:
                with torch.no_grad():
                    outputs = net(imgs)
            if isinstance(outputs,dict):
                seg_logits = outputs['seg_logits']
            else:
                seg_logits = outputs[0]
            seg_logits = F.interpolate(seg_logits,[H,W])
            
            criterion = nn.BCEWithLogitsLoss()
            batch_loss = criterion(seg_logits,true_masks)
            loss += batch_loss.item()
            pred = torch.sigmoid(seg_logits)
            pred = (pred > 0.5).float()
            
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

    net.train()
    iou = TP/(TP+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2*precision*recall/(precision+recall)
    dice = 2*TP/(FP+2*TP+FN)
    test_loss = loss/n_test
    return precision,recall,f1,iou,dice,test_loss