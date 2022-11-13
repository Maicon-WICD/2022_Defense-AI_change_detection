
from datetime import datetime
from time import time
import numpy as np
import shutil, random, os, sys, torch
from glob import glob
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchinfo import summary
from models.DeepLab import *
import torch.optim as optim
import torch.nn as nn
from modules.utils import load_yaml, get_logger
from modules.metrics import get_metric_function
from modules.earlystoppers import EarlyStopper
from modules.losses import get_loss_function
from modules.scalers import get_image_scaler
from modules.datasets import SegDataset
from modules.recorders import Recorder
# from modules.recorders import Recorder
from modules.trainer import train_epoch, eval_epoch
from models.utils import get_model
from modules.losses import loss_adjust_cross_entropy, cross_entropy,loss_adjust_cross_entropy_cdt,get_init_dy, get_init_ly, get_train_w, get_val_w

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

if __name__ == '__main__':
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)

    # Set train serial: ex) 20211004
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_serial = 'debug' if config['debug'] else train_serial

    # Set random seed, deterministic
    torch.cuda.manual_seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create train result directory and set logger
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)
    
    # Set logger
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train',
                        file_path=os.path.join(train_result_dir, 'train.log'),
                        level=logging_level)


    # Set data directory
    train_dirs = os.path.join(prj_dir, 'data', 'train')
    

    # Load data and create dataset for train 
    train_img_paths = glob(os.path.join(train_dirs, 'x', '*.png'))
    
    train_img_paths, eval_img_paths = train_test_split(train_img_paths, test_size=config['eval_size'], random_state=config['seed'], shuffle=True)
    train_img_paths, val_img_paths = train_test_split(train_img_paths, test_size=config['val_size'], random_state=config['seed'], shuffle=True)
    eval_img_paths, eval_val_img_paths = train_test_split(train_img_paths, test_size=config['eval_size'], random_state=config['seed'], shuffle=True)
    train_dataset = SegDataset(paths=train_img_paths,
                            input_size=[config['input_width'], config['input_height']],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    
    val_dataset = SegDataset(paths=val_img_paths,
                            input_size=[config['input_width'], config['input_height']],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    
    eval_dataset = SegDataset(paths=eval_img_paths,
                            input_size=[config['input_width'], config['input_height']],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    
    eval_val_dataset = SegDataset(paths=eval_val_img_paths,
                            input_size=[config['input_width'], config['input_height']],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    # Create data loader
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=6,
                                num_workers=config['num_workers'], 
                                shuffle=config['shuffle'],
                                drop_last=config['drop_last'])

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=6,
                                num_workers=config['num_workers'], 
                                shuffle=False,
                                drop_last=config['drop_last'])

    eval_dataloader = DataLoader(dataset=eval_dataset,
                                batch_size=40,
                                num_workers=config['num_workers'], 
                                shuffle=False,
                                drop_last=config['drop_last'])
    
    eval_val_dataloader = DataLoader(dataset=eval_val_dataset,
                                batch_size=40,
                                num_workers=config['num_workers'], 
                                shuffle=False,
                                drop_last=config['drop_last'])
    
    
    logger.info(f"Load dataset, train: {len(train_dataset)}, val: {len(val_dataset)}")
    
    dy = get_init_dy(config, len(train_dataset))
    ly = get_init_ly(config, len(train_dataset))
    #w_train = get_train_w(config, len(train_dataset))
    #w_val = get_val_w(config, len(val_dataset))

    up_start_epoch=config["up_configs"]["start_epoch"]
    
    def warm_up_with_multistep_lr_low(epoch): return (epoch+1) / config["low_lr_warmup"] \
        if epoch < config["low_lr_warmup"] \
        else 0.1**len([m for m in config["low_lr_schedule"] if m <= epoch])

    def warm_up_with_multistep_lr_up(epoch): return (epoch-up_start_epoch+1) / config["up_lr_warmup"] \
        if epoch-up_start_epoch < config["up_lr_warmup"] \
        else 0.1**len([m for m in config["up_lr_schedule"] if m <= epoch])
        
    # Load model
    if config['architecture'] == "DeepLab_F":
      Net = DeepLab(num_classes=4)
    else:
      Net = get_model(model_str=config['architecture'])
      Net = Net(classes=4,
                  encoder_name=config['encoder'],
                  encoder_weights=config['encoder_weight'],
                  activation=config['activation']).cuda()
    model = nn.DataParallel(Net,device_ids=[0,1])
         
    recorder = Recorder(record_dir=train_result_dir,
                        model=model,
                        logger=logger)
    # summary(model,input_size=(16,3,256,480))
    
    metric_funcs = {metric_name:get_metric_function(metric_name) for metric_name in config['metrics']}
    logger.info(f"Load model architecture: {config['architecture']}")
    
    # Set optimizer
    train_optimizer = optim.SGD(params=model.parameters(),
                            lr=config["low_lr"], momentum=0.9, weight_decay=1e-4)
    val_optimizer = optim.SGD(params=[{'params': dy}, {'params': ly}],
                                lr=config["up_lr"], momentum=0.9, weight_decay=1e-4)
    
    # Set Scheduler
    train_lr_scheduler = optim.lr_scheduler.LambdaLR(train_optimizer, lr_lambda=warm_up_with_multistep_lr_low)
    val_lr_scheduler = optim.lr_scheduler.LambdaLR(val_optimizer, lr_lambda=warm_up_with_multistep_lr_up)
    
    # Set loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("START TRAINING")
    logger.info("START TRAINING")
    for epoch_id in range(config['n_epochs']):
        row = dict()
        row['epoch_id'] = epoch_id
        row['train_serial'] = train_serial

        if epoch_id % config["eval_interval"] == 0:
            if config["up_configs"]["dy_init"]=="CDT":
                print("CDT")
                text, loss, scores = eval_epoch(eval_dataloader, model, loss_adjust_cross_entropy_cdt, epoch_id, ' train_dataset', config, params=[dy, ly],
            metric_funcs=metric_funcs)

            else:
                text, loss, scores = eval_epoch(eval_dataloader, model,
                                                                loss_adjust_cross_entropy, epoch_id, ' train_dataset', config,
                                                                params=[dy, ly])

            text, loss, scores = eval_epoch(eval_val_dataloader, model,
                                                            cross_entropy, epoch_id, ' val_dataset', config, params=[dy, ly],
            metric_funcs=metric_funcs)
            for metric_name, metric_score in scores.items():
                row[f'val_{metric_name}'] = metric_score    
            row['val_loss'] = loss
        loss, scores = train_epoch(epoch_id, model, config,
            low_loader=train_dataloader, low_criterion=loss_adjust_cross_entropy,
            low_optimizer=train_optimizer, low_params=[dy, ly],
            up_loader=val_dataloader, up_optimizer=val_optimizer,
            metric_funcs=metric_funcs,
            up_criterion=cross_entropy, up_params=[dy, ly])
            
        row['train_loss'] = loss
        for metric_name, metric_score in scores.items():
            row[f'train_{metric_name}'] = metric_score
        recorder.add_row(row)
        # Performance record - plot
        recorder.save_plot(config['plot'])
        
        train_lr_scheduler.step()
        val_lr_scheduler.step()
        
    print("END TRAINING")
    logger.info("END TRAINING")