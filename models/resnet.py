# -*- coding: utf-8 -*-
from torchvision.models import resnet18
from torch import nn

class resnet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features,num_classes)
        for params in self.model.parameters():
            params.requires_grad = False
        for params in self.model.layer4.parameters():
            params.requires_grad = True
        for params in self.model.fc.parameters():
            params.requires_grad = True
        
    def forward(self,x):
        x = self.model(x)
        return x

