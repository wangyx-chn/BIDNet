import torch
import torch.nn as nn 
import loss
from loss import KLLoss

class Distiller():
    def __init__(self,config,batch,outputs,device) -> None:
        self.config = config
        self.batch = batch
        self.outputs = outputs
        self.device = device
        pass

    def loss(self):
        names = [item['name'] for item in self.config]
        weights = [item['weight'] for item in self.config]
        preds = []
        targets = []
        functions = []
        for i,name in enumerate(names):
            func_name = self.config[i]['loss']
            target = self.batch[name]
            target = target.to(device=self.device, dtype=torch.float32)
            branch = name.split('_')[0]
            if branch == 'decoder':
                pred = self.outputs['decoder_features']
            else:
                index = eval(name.split('_')[1])
                pred = self.outputs[f'{branch}_features'][index]
            
            if func_name == 'StructLoss':
                edge_gts = self.batch['edgegt']
                edge_gts = edge_gts.to(device=self.device, dtype=torch.float32)
                edge_supmask = (edge_gts!=0)*1
                sim_mat = loss.SimMat()
                target,pred = sim_mat(target,pred,edge_supmask)
                func_name = self.config[i]['func']

            try:
                func = getattr(nn,func_name)
                loss_func = func()
            except:
                loss_func = eval(func_name)()

            preds.append(pred)
            targets.append(target)
            functions.append(loss_func)
        dist_loss,losses = loss.loss_calc(preds,targets,functions,weights)
        losses_dict = {name:losses[i] for i,name in enumerate(names)}
        return dist_loss,losses_dict
    
class DistillerOnline():
    def __init__(self,config,outputs_T,outputs,batch) -> None:
        self.config = config
        self.outputs_T = outputs_T
        self.outputs = outputs
        self.batch = batch
        pass
        
    def loss(self):
        names = [item['name'] for item in self.config]
        weights = [item['weight'] for item in self.config]
        preds = []
        targets = []
        functions = []
        for i,name in enumerate(names):
            func_name = self.config[i]['loss']
            branch = name.split('_')[0]
            if branch == 'decoder':
                pred = self.outputs['decoder_features']
                target = self.outputs_T['decoder_features']
            else:
                index = eval(name.split('_')[1])
                pred = self.outputs[f'{branch}_features'][index]
                target = self.outputs_T[f'{branch}_features'][index]
            
            if func_name == 'StructLoss':
                edge_gts = self.batch['edgegt']
                #edge_gts = edge_gts.to(device=self.device, dtype=torch.float32)
                edge_supmask = (edge_gts!=0)*1
                sim_mat = loss.SimMat()
                target,pred = sim_mat(target,pred,edge_supmask)
                func_name = self.config[i]['func']

            try:
                func = getattr(nn,func_name)
                loss_func = func()
            except:
                loss_func = eval(func_name)()

            preds.append(pred)
            targets.append(target)
            functions.append(loss_func)
        dist_loss,losses = loss.loss_calc(preds,targets,functions,weights)
        losses_dict = {name:losses[i] for i,name in enumerate(names)}
        return dist_loss,losses_dict
    