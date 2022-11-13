"""Trainer
"""

from tqdm import tqdm
import torch
from modules.metrics import topk_corrects
from modules.losses import *
from tqdm import tqdm
from torch.autograd import grad
import numpy as np

def train_epoch(
    cur_epoch, model, args,
    low_loader, low_criterion , low_optimizer, low_params=None,
    up_loader=None, up_optimizer=None, up_criterion=None, up_params=None,
    metric_funcs=None
    ):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    scores = {metric_name: 0 for metric_name, _ in metric_funcs.items()}
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    ARCH_EPOCH=args["up_configs"]["start_epoch"]
    ARCH_END=args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL=args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL=args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE=args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE=args["up_configs"]["val_batches"]
    device=args["device"]
    is_up=(cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
        ((cur_epoch+1) % ARCH_EPOCH_INTERVAL) == 0
    
    if is_up:
        print('lower lr: ',low_optimizer.param_groups[0]['lr'],'  upper lr: ',up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        low_iter_alt=iter(low_loader)
    else:
        print('lr: ',low_optimizer.param_groups[0]['lr'])
    
    model.train()
    total_sample=0.
    total_loss=0.
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 2*((num_classes-1)//group_size)+1
    use_reg=True

    d_train_loss_d_w = torch.zeros(num_weights,device=device)

    for cur_iter, (low_data, low_targets) in enumerate(tqdm(low_loader)):
        low_data, low_targets = low_data.float().to(device=device, non_blocking=True), low_targets.to(device=device, non_blocking=True).long()
        if is_up:
            model.train()
            up_optimizer.zero_grad()
            if cur_iter%ARCH_INTERVAL==0:
                for _ in range(ARCH_TRAIN_SAMPLE):
                    try:
                        low_data_alt, low_targets_alt = next(low_iter_alt)
                    except StopIteration:
                        low_iter_alt = iter(low_loader)
                        low_data_alt, low_targets_alt = next(low_iter_alt) 
                    low_data_alt, low_targets_alt = low_data_alt.to(device=device, non_blocking=True).float(), low_targets_alt.to(device=device, non_blocking=True).long()
                    low_optimizer.zero_grad()
                    low_preds=model(low_data_alt)
                    low_loss=low_criterion(low_preds,low_targets_alt,low_params,group_size=group_size) 
                    d_train_loss_d_w+=gather_flat_grad(grad(low_loss,model.parameters(),create_graph=True))
                d_train_loss_d_w/=ARCH_TRAIN_SAMPLE
                d_val_loss_d_theta = torch.zeros(num_weights,device=device)
              
                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        up_data, up_targets = next(up_iter)
                    except StopIteration:
                        up_iter = iter(up_loader)
                        up_data, up_targets = next(up_iter) 
                    up_data, up_targets = up_data.to(device=device, non_blocking=True).float(), up_targets.to(device=device, non_blocking=True).long()
                    model.zero_grad()
                    low_optimizer.zero_grad()
                    up_preds = model(up_data)
                    up_loss = up_criterion(up_preds,up_targets,up_params,group_size=group_size)
                    d_val_loss_d_theta += gather_flat_grad(grad(up_loss, model.parameters(), retain_graph=use_reg))
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                preconditioner = d_val_loss_d_theta
                
                preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 1.0,
                                                                5, model)
                indirect_grad = gather_flat_grad(
                    grad(d_train_loss_d_w, get_trainable_hyper_params(up_params), grad_outputs=preconditioner.view(-1),allow_unused=True))
                hyper_grad =- indirect_grad
                up_optimizer.zero_grad()
                assign_hyper_gradient(up_params,hyper_grad,num_classes)
                up_optimizer.step()
                d_train_loss_d_w = torch.zeros(num_weights,device=device)
        
        if is_up:
            try:
                up_data, up_targets = next(up_iter)
            except StopIteration:
                up_iter = iter(up_loader)
                up_data, up_targets = next(up_iter) 
            up_data, up_targets = up_data.to(device=device, non_blocking=True).float(), up_targets.to(device=device, non_blocking=True).long()
            up_preds=model(up_data)
            up_loss=up_criterion(up_preds,up_targets,up_params,group_size=group_size)
            up_optimizer.zero_grad()
            up_loss.backward()
            up_optimizer.step()

        low_preds = model(low_data)
        loss = low_criterion(low_preds, low_targets, low_params,group_size=group_size)
        
        low_optimizer.zero_grad()
        loss.backward()
        low_optimizer.step()
        loss = loss.item()
        total_loss+=loss

        # Metric
        for metric_name, metric_func in metric_funcs.items():
            scores[metric_name] += metric_func(low_preds.argmax(1), low_targets).item() / len(low_loader)
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/len(low_loader)}  ')
    return total_loss/len(low_loader), scores

@torch.no_grad()
def eval_epoch(data_loader, model, criterion, cur_epoch, text, config,params=None,metric_funcs=None):
    group_size=config["group_size"]
    model.eval()
    loss=0.
    scores = {metric_name: 0 for metric_name, _ in metric_funcs.items()}
    for cur_iter, (data, targets) in enumerate(tqdm(data_loader)):
        data, targets = data.float().cuda(), targets.cuda(non_blocking=True).long()
        logits = model(data)
        mb_size = data.size(0)
        loss += criterion(logits,targets,params,group_size=group_size).item()*mb_size
    
        for metric_name, metric_func in metric_funcs.items():
            scores[metric_name] += metric_func(logits.argmax(1), targets).item() / len(data_loader)
    text=f'{text}: Epoch {cur_epoch} :  Loss = {loss/len(data_loader)} '
    print(scores)
    return text, loss/(len(data_loader)), scores
