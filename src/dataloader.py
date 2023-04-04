import torch
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


annotations=pd.read_csv("/kaggle/input/dataset-1-and-dataframe/dataframe/key_points.csv")
annotations.volumns=[i.lower() for i in annotations.columns]
annotations.dropna(axis=0,inplace=True)
selected_images=set(annotations['img_name'].values)


data_path_1="/kaggle/input/dataset-1-and-dataframe/dataset_1"
data_path_2="/kaggle/input/dataset-2/dataset_2"

path_1_images=list(set(os.listdir(data_path_1)).intersection(selected_images))
path_2_images=list(set(os.listdir(data_path_2)).intersection(selected_images))


num_train_samples=1200
num_val_samples=len(path_1_images)+len(path_2_images)-num_train_samples 


train_data_list=list(map(lambda x : os.path.join(data_path_1,x),path_1_images))+list(map(lambda x :os.path.join(data_path_2,x),path_2_images[:len(path_2_images)-num_val_samples]))
val_data_list=list(map(lambda x : os.path.join(data_path_2,x),path_2_images[(len(path_2_images)-num_val_samples):]))


class FaceData:
    def __init__(self,data,annotations,transforms=None):
        self.transforms=transforms
        self.data=data
        self.annotations=annotations
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        data=self.data[idx]
        labels=torch.tensor(self.annotations.loc[annotations['img_name']==data[-9:]].iloc[:,1:].values.reshape(-1,2),dtype=torch.float32)
        
        if self.transforms is not None:
            image=Image.open(data)
            image=self.transforms(image)
        
        return {"x":image,'y':labels}
            
image_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5), #can be a hyper param
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2)
])
   
train_data=FaceData(train_data_list,annotations,image_transforms)
val_data=FaceData(val_data_list,annotations,image_transforms)

def data_init(train_batch_size,val_batch_size):
    train_loader=DataLoader(train_data,train_batch_size)
    val_loader=DataLoader(val_data,val_batch_size)
    return train_loader,val_loader