# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from PIL import Image

def visualize_CAM_heatmap(orig_img, cam, alpha=0.5, colormap=cv.COLORMAP_JET):
    if isinstance(orig_img,Image.Image):
        orig_img = np.array(orig_img)
    
    heatmap = cv.applyColorMap(cam,colormap)
    overlay = np.uint8((1-alpha) * orig_img + alpha * heatmap)
    
    return overlay