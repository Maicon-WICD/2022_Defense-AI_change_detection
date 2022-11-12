import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from utils import initialize_metrics, get_mean_metrics, set_metrics, hybrid_loss
# from sklearn.metrics import precision_recall_fscore_support as prfs
from models.Models import Siam_NestedUNet_Conc, SNUNet_ECAM
from datasets.dataset import get_loaders
from utils import get_metric_function, hybrid_loss
from loss import CCE, GeneralizedDiceLoss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
metrics_name = ['miou', 'iou1', 'iou2', 'iou3']
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    seed_everything(args.seed)
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_loader, val_loader = get_loaders(args) ####### 메모
    
    # model =                  ######## 모델 메모
    model = Siam_NestedUNet_Conc(in_ch=args.input_channel, out_ch=args.output_channel).to(device)
    
    criterion = hybrid_loss ## get_criterion 메모
#     criterion = GeneralizedDiceLoss()
#     criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    
    
    # train 시퀀스 메모
#     best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1, 'cd_miou':-1}
    best_metrics = {metric_name:-1 for metric_name in metrics_name}
    metric_funcs = {metric_name:get_metric_function(metric_name) for metric_name in metrics_name}
    
    total_step = -1
#     evaluator = Evaluator(args.output_channel) # 추가 - num_class = 2 metadata 에 추가해야함
    
    early_stop_count = 0 ###########

    for epoch in range(args.epochs):
        train_metrics = {metric_name:0 for metric_name in metrics_name}
        val_metrics = {metric_name:0 for metric_name in metrics_name}

        model.train()
        
        batch_iter = 0
        tbar = tqdm(train_loader)
        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+args.batch_size))
            batch_iter = batch_iter+args.batch_size
            total_step += 1
            # Set variables for training
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.long().to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Get model predictions, calculate loss, backprop
            cd_preds = model(batch_img1, batch_img2)
#             print(cd_preds.shape)
            cd_preds = cd_preds[-1]
            cd_loss = criterion(cd_preds, labels)
            loss = cd_loss
            
#             print(loss.item())
            loss.backward()
            optimizer.step()
            
#             print(loss)
            # print(cd_preds.argmax(1))
            # print(labels)
            
            
            # Metric
            for metric_name, metric_func in metric_funcs.items():
                train_metrics[metric_name] += metric_func(cd_preds.argmax(1), labels).item() / len(train_loader)
                # print(metric_func(cd_preds.argmax(1), labels).item())
                # print(metric_func(cd_preds.argmax(1), labels).item() / len(train_loader))
            
            # Calculate and log other batch metrics
#             cd_corrects = (100 *
#                            (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
#                            (labels.size()[0] * (args.patch_size**2)))

#             cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
#                                    cd_preds.data.cpu().numpy().flatten(),
#                                    average='macro',
#                                    zero_division=0,
#                                    pos_label=1)
            
#             train_metrics = set_metrics(train_metrics,
#                                     cd_loss,
#                                     cd_corrects,
#                                     cd_train_report,
#                                     scheduler.get_last_lr())
            
#             mean_train_metrics = get_mean_metrics(train_metrics)
            
            # clear batch variables from memory
            del batch_img1, batch_img2, labels
        scheduler.step()
#         print("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))
        print("EPOCH {} TRAIN METRICS".format(epoch) + str(train_metrics) + "|| loss :"+str(loss.item()))
        
        # valid
        model.eval()
#         evaluator.reset()
        
        with torch.no_grad():
            for batch_img1, batch_img2, labels in val_loader:
                # Set variables for training
                batch_img1 = batch_img1.float().to(device)
                batch_img2 = batch_img2.float().to(device)
                labels = labels.long().to(device)

                # Get predictions and calculate loss
                cd_preds = model(batch_img1, batch_img2)
                cd_preds = cd_preds[-1]
                cd_loss = criterion(cd_preds, labels)

                
                for metric_name, metric_func in metric_funcs.items():
                    val_metrics[metric_name] += metric_func(cd_preds.argmax(1), labels).item() / len(val_loader)
                    # print(metric_func(cd_preds.argmax(1), labels).item(), len(val_loader))
                    # print(metric_func(cd_preds.argmax(1), labels).item() / len(val_loader))
#                 _, cd_preds = torch.max(cd_preds, 1)

                # Calculate and log other batch metrics
#                 cd_corrects = (100 *
#                                (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
#                                (labels.size()[0] * (args.patch_size**2)))

#                 cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
#                                      cd_preds.data.cpu().numpy().flatten(),
#                                      average='macro',
#                                      zero_division=0,
#                                      pos_label=1)

#                 val_metrics = set_metrics(val_metrics,
#                                           cd_loss,
#                                           cd_corrects,
#                                           cd_val_report,
#                                           scheduler.get_last_lr())

                # log the batch mean metrics
#                 mean_val_metrics = get_mean_metrics(val_metrics)
#                 evaluator.add_batch(labels, cd_preds) ##########

                # clear batch variables from memory
                del batch_img1, batch_img2, labels
                
#             mIoU = evaluator.Mean_Intersection_over_Union() # 추가
#             mean_val_metrics['cd_miou'] = mIoU # 추가
            
#             print("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))
            print("EPOCH {} VALIDATION METRICS".format(epoch)+str(val_metrics))
            
            
#             if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
#                 or
#                 (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
#                 or
#                 (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):
            if val_metrics['miou'] > best_metrics['miou']:
                print("update model")
                
                # Save model
                if not os.path.exists(f'./{args.name}'):
                    os.mkdir(f'./{args.name}')
                
                torch.save(model, f"./{args.name}/checkpoint_{str(epoch)}_{val_metrics['miou']:4.4}.pt")
                best_metrics = val_metrics
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= args.patience:
                print(f'--------epoch {epoch} early stopping--------')
                print(f'--------epoch {epoch} early stopping--------')
                break
            print()
                                      
                
    ###################################################### 메모

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset_path', type=str, help='dataset path')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--patch_size', type=int, default=256, help='input patch size (default: 256)')
    
    parser.add_argument('--input_channel', type=int, default=3, help='input channel (default: 3)')
    parser.add_argument('--output_channel', type=int, default=2, help='output channel (num_class) (default: 2)')
    parser.add_argument('--patience', type=int, default=11, help='early stop (default: 11)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio (default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--step_size', type=int, default=10, help='stepLR step size')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    args = parser.parse_args()
    print(args)
    
    train(args)
    
    
