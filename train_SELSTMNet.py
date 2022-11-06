import os
import numpy as np
import nibabel as nib
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable 

from sklearn.metrics import cohen_kappa_score

from datetime import datetime
from time import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=str, default='0', help='GPU node num')
parser.add_argument('--fold', type=int, default=1, help='fold num (1~5)')
parser.add_argument('--img_PATH', type=str, default='/workspace/Conference/2022_KIICE(한국정보통신학회 추계)/dataset/gt_seg_polar/train/cls_img/', help='input image path')
parser.add_argument('--label_path', type=str, default='slice_label.json', help='label path')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset_training(Dataset):
    def __init__(self, dataset):
        self.img_list = [path for path in dataset["img"]]
        self.gt_list = [gt for gt in dataset["gt"]]
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = nib.load(img).get_fdata()
        gt = int(self.gt_list[idx])
        
        # rotation
        start = random.randrange(0,90)
        img = np.concatenate([img[start:],img[:start]],axis=0)

        # vertical flip
        if random.random() > 0.5:
            img = np.flip(img, 0)
                    
        img = torch.Tensor(img.copy())
        
        return img, gt

class Dataset_testing(Dataset):
    def __init__(self, dataset):
        self.img_list = [path for path in dataset["img"]]
        self.gt_list = [gt for gt in dataset["gt"]]
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = nib.load(img).get_fdata()
        gt = int(self.gt_list[idx])
        
        #### TTA ####
        # rotation
        start = random.randrange(0,90)
        img = np.concatenate([img[start:],img[:start]],axis=0)

        # vertical flip
        if random.random() > 0.5:
            img = np.flip(img, 0)
                    
        img = torch.Tensor(img.copy())
        
        return img, gt

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
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
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

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def main():
    fold_path = "/workspace/Challenge/COSMOS2022/fold_idx_full/"

    test_path = fold_path+"fold"+str(args.fold)+"_testing.npy"
    pat_test = np.load(test_path)
    train_path = fold_path+"fold"+str(args.fold)+"_training.npy"
    pat_train = np.load(train_path)

    label_data = 0
    with open(args.label_path) as json_file:
        label_data = json.load(json_file)

    train_dataset = dict()
    train_dataset["img"], train_dataset["gt"] = [], []
    test_dataset = dict()
    test_dataset["img"], test_dataset["gt"] = [], []

    slices = [i.split(".")[0] for i in os.listdir(args.img_PATH)]
    for s in slices:
        pat = s.split("_")[0]
        artery = s.split("_")[1]
        s_num = s.split("_")[2]
        
        if pat in pat_test:
            if s_num in label_data[pat][artery]:
                test_dataset["img"].append(args.img_PATH+s+".nii.gz")
                test_dataset["gt"].append(int(label_data[pat][artery][s_num]))
        elif pat in pat_train:
            if s_num in label_data[pat][artery]:
                train_dataset["img"].append(args.img_PATH+s+".nii.gz")
                train_dataset["gt"].append(int(label_data[pat][artery][s_num]))

    print("Num of train dataset :", len(train_dataset["img"]))
    print("Num of test dataset :", len(test_dataset["img"]))

    trainset = Dataset_training(train_dataset)
    testset = Dataset_testing(test_dataset)

    training_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=10)
    validation_loader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=10)

    model = SE_LSTMNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
   
    torch.autograd.set_detect_anomaly(False)
    def train_one_epoch(epoch_index):
        start = time()
        running_loss = 0.
        last_loss = 0.
        
        running_acc = 0.
        last_acc = 0.
        
        for i, data in enumerate(training_loader):        
            inputs, labels = data
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                inputs, labels = inputs.float().to(device), labels.float().to(device)
                labels = labels.view(-1,1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = binary_acc(outputs, labels)
            
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ')
                exit(1)
            
            scaler.scale(loss).backward()
            # loss.backward()
            
            scaler.step(optimizer)
            # scaler.step(scheduler)
            
            scaler.update()
            
            running_loss += loss.item()
            running_acc += acc.item()
            
            
            if i % 50 == 49:
                last_loss = running_loss / 50
                last_acc = running_acc / 50
                print('batch {} loss: {}, acc: {}'.format(i+1, last_loss, last_acc))
                running_loss = 0.
                running_acc = 0.
                
                
        if epoch_index % 10 == 9:
            optimizer.param_groups[0]['lr'] *= 0.5
        
            
        end = time()
        print("epoch", epoch_index, end-start)

        return last_loss, last_acc

    if not os.path.isdir("cls_SELSTMNet"):
        os.mkdir("cls_SELSTMNet")

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    loss_txt = open("cls_SELSTMNet/fold"+str(args.fold)+"_LSTM+SE_masked_{}.txt".format(timestamp), "a+")

    EPOCHS = 1000

    best_kappa = 0.0
    best_vloss = 1000000

    for epoch in range(EPOCHS):
        
    #     gc.collect()
    #     torch.cuda.empty_cache()
        
        print('EPOCH {}:'.format(epoch_number + 1))
        
        model.train(True)
        avg_loss, avg_acc = train_one_epoch(epoch_number)
        
        model.train(False)
        running_vloss = 0.0
        running_vacc = 0.0
        val_kappa = []
        
        for i, vdata in enumerate(validation_loader):
            with torch.no_grad():
                vinputs, vlabels = vdata
                vlabels = vlabels.view(-1,1)
                with torch.cuda.amp.autocast():
                    vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)
                    voutputs = model(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    vacc = binary_acc(voutputs, vlabels)
                running_vloss += vloss.item()
                running_vacc += vacc.item()
                                
                y_actual = vlabels.data.cpu().numpy()
                voutputs = torch.sigmoid(voutputs)
                y_pred = voutputs[:,-1].detach().cpu().numpy()
                val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i+1)
        avg_kappa = np.mean(val_kappa)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        
        # Log the running loss averaged per batch
        # for both training and validation
        loss_txt.write('EPOCH {}: LOSS train loss {}, train acc {}, valid loss {}, valid acc {}, val kappa {}\n'.format(epoch_number+1, avg_loss, avg_acc, avg_vloss, avg_vacc, avg_kappa))
        loss_txt.flush()
            
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'cls_SELSTMNet/fold'+str(args.fold)+'_best_model_{}_{}epoch.pth'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
        else:
            model_path = 'cls_SELSTMNet/fold'+str(args.fold)+'_last_model_{}.pth'.format(timestamp)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
    loss_txt.close()


if __name__ == "__main__":
    main()
