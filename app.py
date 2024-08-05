import h5py
import numpy as np
import pandas as pd
import dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, auc
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.transforms import v2
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler



@torch.enable_grad
def train(model, loader, criterion, opt):
    model.train()
    errors = []
    
    device = next(model.parameters()).device
    for x, y in  tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
                
        loss = criterion(out.squeeze(), y.float())

        opt.zero_grad()
        loss.backward()
        opt.step()
        errors.append(loss.item())
        
    return sum(errors) / len(errors)



@torch.no_grad
def test(model, loader):
    model.eval()
    #accs = []
    paucs = []
    
    device = next(model.parameters()).device
    for x, y in tqdm(loader):
        x = x.to(device)
       # y = y.to(device)
        out = model(x)
        probs = torch.sigmoid(out)
        #if torch.any(y):
        fpr, tpr, _ = roc_curve(y, probs.cpu())
        paucs.append(auc(fpr, tpr))
                    
    return sum(paucs) / len(paucs)
    
        
        
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 1)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        return x
        


if __name__ == "__main__":
    df_meta_train = pd.read_csv("data\\train-metadata.csv")
    df_meta_test = pd.read_csv("data\\test-metadata.csv")
    print(df_meta_train.head())
    
    train_X, test_X = train_test_split(df_meta_train, test_size=0.2, stratify=df_meta_train["target"]) 
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    print(train_X["target"].sum(), test_X["target"].sum())
    
    

    tf_train = v2.Compose([v2.ToImage(), v2.CenterCrop(size=120), v2.RandomHorizontalFlip(), v2.RandomRotation(180), 
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    tf_test = v2.Compose([v2.ToImage(), v2.Resize(120, antialias=True) , v2.ToDtype(torch.float32, scale=True), 
                      v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    ds_train = dataset.CustomDataset(train_X, tf=tf_train)
    ds_test = dataset.CustomDataset(test_X, tf=tf_test)

    freq_c1 = train_X["target"].sum()
    freq_c2 = len(train_X) - train_X["target"].sum()
    print(freq_c1, freq_c2)

    w_train = []
    print(len(train_X), freq_c1+freq_c2)
    for target in train_X["target"]:
        if target == 0:
            w_train.append(1/freq_c2)
        else:
            w_train.append(1/freq_c1)
            
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    sampler_train = WeightedRandomSampler(weights=w_train, num_samples=len(ds_train), replacement=True)
    train_loader = DataLoader(ds_train, sampler=sampler_train, batch_size=32, num_workers=4)
    test_loader = DataLoader(ds_test, batch_size=512, num_workers=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    n_epochs = 5

    model = Model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(n_epochs):
        err = train(model, train_loader, criterion=criterion, opt=opt)
        print("Error")
        print(err)
        a = test(model, test_loader)
        print("Auc")
        print(a)
        