import torch 
import numpy as np
from config import config
from dataset import HumanDataset
#from models.model import*
import pandas as pd 
from tqdm import tqdm 
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
import os 
from torch.utils.data import DataLoader
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')



def test(test_loader,model,folds):
    sample_submission_df = pd.read_csv("csv/sample_submission.csv")
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.cuda()
    model.eval()
    submit_results = []
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            #print(label > 0.5)
           
            labels.append(label > 0.15)
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_bestloss_submission.csv'%config.model_name, index=None)

if __name__ == '__main__':
    test_files = pd.read_csv("csv/sample_submission.csv")
    #model = get_net()
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._conv_stem = nn.Conv2d(config.channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 28)
    model.cuda()
    fold = 0
    test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=0)
    best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
        #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold)
