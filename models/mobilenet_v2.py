# -*- coding: utf-8 -*-
from torchvision.models import mobilenet_v2
from torch import nn

class mobilenet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = mobilenet_v2(True,num_classes=1000)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features,num_classes)
        for params in self.model.parameters():
            params.requires_grad = False
        for params in self.model.features[18].parameters():
            params.requires_grad = True
        for params in self.model.classifier.parameters():
            params.requires_grad = True
        
    def forward(self,x):
        x = self.model(x)
        return x