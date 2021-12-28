import os
import pandas as pd
import sys 
import shutil
from config import config
import torch
import glob
# save best model
def tmp():
    new_file_list = []
    external_file = pd.read_csv("csv/external.csv")
    a = external_file.values.tolist()
    num = len(a)
    for i in range(num):
        csv_name = external_file["Id"][i]
        image_name = 'data/train/' + csv_name 
        if os.path.isfile(image_name + '_red.png') and os.path.isfile(image_name + '_blue.png') and os.path.isfile(image_name + '_green.png') and os.path.isfile(image_name + '_yellow.png'):
            new_file_list.append(a[i])
    name = ['Id', 'Target']  
     
    test=pd.DataFrame(columns=name,data=new_file_list)#数据有三列，列名分别为one,two,three
    print(test)
    test.to_csv('csv/mergered.csv',index=False)
    csv_list = glob.glob('csv/tmp/*.csv') #查看同文件夹下的csv文件数
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')
    for i in csv_list: #循环读取同文件夹下的csv文件
        fr = open(i,'rb').read()
        with open('csv/tmp/result.csv','ab') as f: #将结果保存为result.csv
            f.write(fr)
    print('合并完毕！')
    new_file_list = []

def save_checkpoint(state, is_best_loss,is_best_f1,fold):
    filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    if is_best_f1:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass