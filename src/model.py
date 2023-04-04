# model
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as sch

class KeypointModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1=nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,bias=True)
  
        self.relu1=nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,bias=True)
  
        self.relu2=nn.ReLU(inplace=True)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=True)
  
        self.relu3=nn.ReLU(inplace=True)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1=nn.Linear(128*64*64,512,bias=True)
        self.relu4=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(512,67*2,bias=True)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        
        x=self.conv3(x)
        x=self.relu3(x)
        x=self.pool3(x)
        
        x=x.view((-1,128*64*64))
        x=self.fc1(x)
        x=self.relu4(x)
        x=self.fc2(x)
        x=x.view(-1,67,2)
        
        return x
    
def model_init():
    model=KeypointModel()
    model=model.to(device=Config.device)
    return model

def criterion_init(loss):
    if loss=='l2':
        criterion=nn.MSELoss()
    elif loss=='l1':
        criterion=nn.L1Loss()
    return criterion
        

def optimizer_init(model,optim,learning_rate,momentum):
    if optim=='adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    elif optim=='sgd':
        optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    elif optim=='adamw':
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.001)
    return optimizer


def scheduler_init(kind,optimizer,train_loader,epochs,):
    if kind=='cyclic':
        scheduler=sch.CyclicLR(optimizer=optimizer,base_lr=10**-10,max_lr=0.1,step_size_up=(epochs*len(train_loader)/2))
    elif kind=='sgd_wr':
        scheduler=sch.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=epochs*20,T_mult=2,eta_min=10**-10)
    elif kind=='one_cycle':
        scheduler=sch.OneCycleLR(optimizer=optimizer,max_lr=0.1,total_steps=epochs*len(train_loader))
    elif kind=="rop":
        scheduler=sch.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.1,patience=2)
    return scheduler