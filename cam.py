# -*- coding: utf-8 -*-
import datasets, models, config, utils, cam
from PIL import Image
import torch
from torchvision import transforms
import os

# Eigen-CAM
flags = config.flags
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

model = models.mobilenet(num_classes=100)
model_path = os.path.join(flags['model_root'],'model_state_dict.pt')
#model.load_state_dict(torch.load(model_path))
eigencam = cam.eigenCAM(model,'cpu','model.features.18.0',transform)
img = Image.open('n01669191_46.JPEG')
heatmap = eigencam.get_heatmap(img)
overlay = utils.visualize_CAM_heatmap(img,heatmap,alpha=0.4)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(overlay)
plt.title('CAM')
plt.show()

import cv2
cv2.imwrite('cam_output.jpg',overlay)