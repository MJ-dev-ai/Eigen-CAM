# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
import cv2

class eigenCAM:
    def __init__(self,model,device,layer_name,transform):
        super().__init__()
        self.model = model.eval().to(device)
        self.layer_name = layer_name
        self.device = device
        self.transform = transform
        
        self.feature = {}
        
        self._register_hook()
        
    
    def _forward_hook(self,module,x,y):
        self.feature['output'] = y
        
    def _register_hook(self):
        for name,module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(self._forward_hook)
                break
        else:
            ValueError(f'There is no module named {self.layer_name} in the model.\n')
    
    def get_heatmap(self,img):
        with torch.no_grad():
            img_tensor = self.transform(img).to(self.device)
            img_tensor = img_tensor.unsqueeze(0) # Set batch size as 1
            output = self.model(img_tensor)
            feature = self.feature['output']
            
            _,_,VT = torch.linalg.svd(feature)
            V1 = VT[:,:,0,:].unsqueeze(2)
            cam = feature @ V1.repeat(1,1,V1.shape[3],1)
            cam = cam.sum(1) # Channel sum
            cam -= cam.min()
            cam = cam / cam.max() * 255
            cam = cam.cpu().numpy().transpose(1,2,0).astype(np.uint8)
            cam = cv2.resize(cam,dsize=img.size)
        return cam