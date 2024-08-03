
import cv2 as cv
import os
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, tf, path="data\\train-image\\image\\"):
        self.df = df
        self.path = path
        self.tf = tf
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        rec = self.df.iloc[index]
        y = rec["target"]
        file_name = rec["isic_id"] + ".jpg"
        file_name = os.path.join(self.path, file_name)
    
        x = cv.imread(file_name)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = self.tf(x)  
        return x, y

