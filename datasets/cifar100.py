# -*- coding: utf-8 -*-
import os
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader, random_split

class cifar100(Dataset):
    def __init__(self,root='./data',train=True,download=True,transform=None):
        subfolder = 'train' if train else 'test'
        self.root = os.path.join(root,subfolder)
        self.cifar100 = CIFAR100(root=self.root,train=train,download=download)
        self.transform = transform
    
    def __len__(self):
        return len(self.cifar100)
    
    def __getitem__(self,idx):
        img, label = self.cifar100[idx]
        if self.transform:
            img = self.transform(img)
        
        return img, label

def get_dataloader(root,transform,batch_size):
    train_dataset = cifar100(root,train=True,transform=transform)
    test_dataset = cifar100(root,train=False,transform=transform)
    
    train_val_ratio = 0.9
    train_size = int(train_val_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset,val_dataset = random_split(train_dataset,[train_size,val_size])
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader, val_loader, test_loader