import os
import numpy as np
import nibabel as nib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable 

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

import argparse

parser = argparse.ArgumentParser()

class PolarDataset_predicting(Dataset):
    def __init__(self, dataset):
        self.img_list = [path for path in dataset["img"]]
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = nib.load(img).get_fdata()
        
        return img

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=32, num_layers=1):
        super().__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        out = F.relu(hn)
        
        return out
    
class SE_LSTMNet(nn.Module):
    def init_kernel(m):
        if isinstance(m, nn.Conv2d): 
            # Initialize kernels of Conv2d layers as kaiming normal
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # Initialize biases of Conv2d layers at 0
            nn.init.zeros_(m.bias)
    
    def __init__(self):
        super(SE_LSTMNet, self).__init__()
        
        
        self.lstm = LSTM(input_size=32, hidden_size=32, num_layers=1)
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(1,3), padding='same')
        self.conv1_bn = nn.InstanceNorm2d(64)
        self.se1 = SEBlock(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,3), padding='same')
        self.conv2_bn = nn.InstanceNorm2d(128)
        self.se2 = SEBlock(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), padding='same')
        self.conv3_bn = nn.InstanceNorm2d(256)
        self.f = nn.Flatten()
        self.fc1 = nn.Linear(32, 1)
        
        self.dropout1 = nn.Dropout(0.25)
        
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)    
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize kernels of Conv2d layers as kaiming normal
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.dropout1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.se1(x) * x
        x = F.max_pool2d(x, (1,2))
        x = self.dropout1(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.se2(x) * x
        x = F.max_pool2d(x, (1,2))
        x = self.dropout1(F.relu(self.conv3_bn(self.conv3(x))))
        x = F.adaptive_avg_pool2d(x, (32, 1))
        x = x[:,:,:,0]
        y = self.lstm(x)
        y = self.f(y)
        out = self.fc1(y)

        return out
    
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def to_tensor(arr):
    result = arr[np.newaxis,:,:,:]
    return torch.tensor(result.copy(), dtype=torch.float32)

def sensitivity_score(y, pred):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y, pred)
    sensitivity = cm[0,0] / (cm[0,0]+cm[0,1])
    return sensitivity
def specificity_score(y, pred):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y, pred)
    specificity = cm[1,1] / (cm[1,1]+cm[1,0])
    return specificity


def main():
    with open('test_label.json') as f:
        gt = json.load(f)

    model1 = SE_LSTMNet()
    model1.load_state_dict(torch.load("cls_SELSTMNet/fold1_best_model_20220928_022839_14epoch.pth"))

    model2 = SE_LSTMNet()
    model2.load_state_dict(torch.load("cls_SELSTMNet/fold2_best_model_20220928_023908_16epoch.pth"))

    model3 = SE_LSTMNet()
    model3.load_state_dict(torch.load("cls_SELSTMNet/fold3_best_model_20220928_025136_20epoch.pth"))

    model4 = SE_LSTMNet()
    model4.load_state_dict(torch.load("cls_SELSTMNet/fold4_best_model_20220928_030200_29epoch.pth"))

    model5 = SE_LSTMNet()
    model5.load_state_dict(torch.load("cls_SELSTMNet/fold5_best_model_20220928_031208_7epoch.pth"))

    PATH = "/workspace/Dataset/COSMOS2022/Testing_dataset_nifti/"
    pats = os.listdir(PATH)
    img_PATH = "/workspace/Conference/2022_KIICE(한국정보통신학회 추계)/dataset/gt_seg_polar/test/cls_img/"

    results = dict()
    for p in pats:
        results[p] = {"R":{}, "L":{}}
        
    results1 = dict()
    for p in pats:
        results1[p] = {"R":{}, "L":{}}
        
    results2 = dict()
    for p in pats:
        results2[p] = {"R":{}, "L":{}}
        
    results3 = dict()
    for p in pats:
        results3[p] = {"R":{}, "L":{}}
        
    results4 = dict()
    for p in pats:
        results4[p] = {"R":{}, "L":{}}
        
    results5 = dict()
    for p in pats:
        results5[p] = {"R":{}, "L":{}}
        
    imgs = os.listdir(img_PATH)

    for img in imgs:
        img_path = img_PATH+img
        img = img.split(".")[0]
        pat = img.split("_")[0]
        a = img.split("_")[1]
        s = img.split("_")[2]
        
        polar = nib.load(img_path).get_fdata()
        
        pred1 = model1(to_tensor(polar))
        pred1 = pred1.detach().numpy()
        pred1 = pred1[0][0]
        
        pred2 = model2(to_tensor(polar))
        pred2 = pred2.detach().numpy()
        pred2 = pred2[0][0]
        
        pred3 = model3(to_tensor(polar))
        pred3 = pred3.detach().numpy()
        pred3 = pred3[0][0]
        
        pred4 = model4(to_tensor(polar))
        pred4 = pred4.detach().numpy()
        pred4 = pred4[0][0]
        
        pred5 = model5(to_tensor(polar))
        pred5 = pred5.detach().numpy()
        pred5 = pred5[0][0]
        
        ensemble = (pred1+pred2+pred3+pred4+pred5) / 5
        
        pred = sigmoid(ensemble)
        pred1 = sigmoid(pred1)
        pred2 = sigmoid(pred2)
        pred3 = sigmoid(pred3)
        pred4 = sigmoid(pred4)
        pred5 = sigmoid(pred5)
        
        # print(pat,a,s,round(pred))
        results[pat][a].update({str(s):pred})
        results1[pat][a].update({str(s):pred1})
        results2[pat][a].update({str(s):pred2})
        results3[pat][a].update({str(s):pred3})
        results4[pat][a].update({str(s):pred4})
        results5[pat][a].update({str(s):pred5})

    target = []
    pred = []
    pred1 = []
    pred2 = []
    pred3 = []
    pred4 = []
    pred5 = []

    for p in gt:
        for a in gt[p]:
            for s in gt[p][a]:
                if (s in results[p][a]) & (s in gt[p][a]):
                    # print(gt[p][a][s], results[p][a][s])
                    target.append(gt[p][a][s])
                    pred.append(results[p][a][s])
                    pred1.append(results1[p][a][s])
                    pred2.append(results2[p][a][s])
                    pred3.append(results3[p][a][s])
                    pred4.append(results4[p][a][s])
                    pred5.append(results5[p][a][s])

    print("#Accuracy#")
    print("ensemble:",accuracy_score(target, [round(a) for a in pred]))
    print("fold#1:",accuracy_score(target, [round(a) for a in pred1]))
    print("fold#2:",accuracy_score(target, [round(a) for a in pred2]))
    print("fold#3:",accuracy_score(target, [round(a) for a in pred3]))
    print("fold#4:",accuracy_score(target, [round(a) for a in pred4]))
    print("fold#5:",accuracy_score(target, [round(a) for a in pred5]))
    print("\n#F1 score#")
    print("ensemble:",f1_score(target, [round(a) for a in pred]))
    print("fold#1:",f1_score(target, [round(a) for a in pred1]))
    print("fold#2:",f1_score(target, [round(a) for a in pred2]))
    print("fold#3:",f1_score(target, [round(a) for a in pred3]))
    print("fold#4:",f1_score(target, [round(a) for a in pred4]))
    print("fold#5:",f1_score(target, [round(a) for a in pred5]))
    print("\n#Specificity#")
    print("ensemble:",specificity_score(target, [round(a) for a in pred]))
    print("fold#1:",specificity_score(target, [round(a) for a in pred1]))
    print("fold#2:",specificity_score(target, [round(a) for a in pred2]))
    print("fold#3:",specificity_score(target, [round(a) for a in pred3]))
    print("fold#4:",specificity_score(target, [round(a) for a in pred4]))
    print("fold#5:",specificity_score(target, [round(a) for a in pred5]))
    print("\n#Sensitivity#")
    print("ensemble:",sensitivity_score(target, [round(a) for a in pred]))
    print("fold#1:",sensitivity_score(target, [round(a) for a in pred1]))
    print("fold#2:",sensitivity_score(target, [round(a) for a in pred2]))
    print("fold#3:",sensitivity_score(target, [round(a) for a in pred3]))
    print("fold#4:",sensitivity_score(target, [round(a) for a in pred4]))
    print("fold#5:",sensitivity_score(target, [round(a) for a in pred5]))
    print("\n#Cohen's kappa#")
    print("ensemble:",cohen_kappa_score(target, [round(a) for a in pred]))
    print("fold#1:",cohen_kappa_score(target, [round(a) for a in pred1]))
    print("fold#2:",cohen_kappa_score(target, [round(a) for a in pred2]))
    print("fold#3:",cohen_kappa_score(target, [round(a) for a in pred3]))
    print("fold#4:",cohen_kappa_score(target, [round(a) for a in pred4]))
    print("fold#5:",cohen_kappa_score(target, [round(a) for a in pred5]))
    print("\n#AUC#")
    print("ensemble:", roc_auc_score(target, pred))
    print("fold#1:",roc_auc_score(target, pred1))
    print("fold#2:",roc_auc_score(target, pred2))
    print("fold#3:",roc_auc_score(target, pred3))
    print("fold#4:",roc_auc_score(target, pred4))
    print("fold#5:",roc_auc_score(target, pred5))

if __name__ == "__main__":
    main()
