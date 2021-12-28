import os
import time
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from timeit import default_timer as timer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from efficientnet_pytorch import EfficientNet
from torch.nn import MultiLabelSoftMarginLoss
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import HumanDataset
from config import config
from utils import *

import warnings
warnings.filterwarnings('ignore')


fold = 0
#log
 # 4.1 mkdirs
if not os.path.exists(config.submit):
    os.makedirs(config.submit)
if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
    os.makedirs(config.weights + config.model_name + os.sep +str(fold))
if not os.path.exists(config.best_models):
    os.mkdir(config.best_models)
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

data_path = 'data/train'
train_files = pd.read_csv("csv/train.csv")
external_files = pd.read_csv("csv/result.csv")

train_data_list,val_data_list = train_test_split(external_files,test_size = 0.13,random_state = 2050)
#train data
train_gen = HumanDataset(train_data_list, data_path, mode='train')
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=0)
#val data
val_gen = HumanDataset(val_data_list,config.train_data,augument=False,mode="train")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=0)

# model param
model = EfficientNet.from_pretrained('efficientnet-b0')
model._conv_stem = nn.Conv2d(config.channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, 28)
model.cuda()

#criterion = MultiLabelSoftMarginLoss()
criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)


#start train
start = timer()

best_results = [np.inf,0]
val_metrics = [np.inf,0]

for epoch in range(0,config.epochs):
    scheduler.step(epoch)

    lr = get_learning_rate(optimizer)
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()

    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

        #compute output
        output = model(images)
        loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))

        f1_batch = f1_score(target.cpu(),output.sigmoid().cpu() > 0.15,average='macro')
        f1.update(f1_batch,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, f1.avg, 
                val_metrics[0], val_metrics[1], 
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
        log.write("\n")

        train_loss = losses.avg,f1.avg
    
    #val
    losses = AverageMeter()
    f1 = AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            output = model(images_var)
            loss = criterion(output,target)
            losses.update(loss.item(),images_var.size(0))
            f1_batch = f1_score(target.cpu(),output.sigmoid().cpu().data.numpy() > 0.15,average='macro')
            f1.update(f1_batch,images_var.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    train_loss[0], train_loss[1], 
                    losses.avg, f1.avg,
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))

            print(message, end='',flush=True)
        log.write("\n")
        val_metrics = losses.avg,f1.avg
    
    # check results 
    is_best_loss = val_metrics[0] < best_results[0]
    best_results[0] = min(val_metrics[0],best_results[0])
    is_best_f1 = val_metrics[1] > best_results[1]
    best_results[1] = max(val_metrics[1],best_results[1])   

    # save model
    save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_loss":best_results[0],
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "best_f1":best_results[1],
        },is_best_loss,is_best_f1,fold)

    # print logs
    print('\r',end='',flush=True)
    log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "best", epoch, epoch,                    
                train_loss[0], train_loss[1], 
                val_metrics[0], val_metrics[1],
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
            )
    log.write("\n")
    time.sleep(0.01)