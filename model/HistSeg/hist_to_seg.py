import os
import sys
import cv2
import torch
import argparse
import torch as t
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial.distance import dice
import SimpleITK as sitk
import time


from metric import dice_coef_multilabel
from dilated_unet import Segmentation_model
from dataset import ImageProcessor, DataGenerator
from utils import soft_to_hard_pred, keep_largest_connected_components


def hist_to_seg(unet_model,img_fr):
    w, h, c = img_fr.shape
    
    img_input = cv2.resize(img_fr,(256,256))/255.0

    image = np.expand_dims(img_input.transpose((2,0,1)),0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = torch.Tensor(image.astype(np.float32)).to(device)
    prediction = unet_model(image)
    y_pred = 255*np.argmax(soft_to_hard_pred(prediction[0].cpu().detach().numpy(), 1), axis=1)[0]

    hist_seg = np.zeros((256,256,3))
    hist_seg[:,:,0] = y_pred 
    hist_seg[:,:,1] = y_pred 
    hist_seg[:,:,2] = y_pred 

    
    hist_seg = cv2.resize(hist_seg,(h,w))
    
    return hist_seg
