from __future__ import print_function

import os
import sys
from argparse import ArgumentParser
from time import time

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from tensorflow.keras.optimizers import Adam
import SimpleITK as sitk
from scipy import ndimage, misc
import glob

sys.path.append('model/GcGAN/')
from models.gc_cycle_gan_model import *

sys.path.append('model/RegNet/network/')
from model import *
from image_reader import *
from loss import *
from transform import *
from combine_transforms import *
from tps import *


def NCC(mri,hist):
    a  = mri[:,:,:,:]
    b = hist[:,:,:,:]
    a = a.flatten()
    b = b.flatten()
    
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
        
    return np.correlate(a, b)


def get_slice_correspondence(model_aff, model_tps, netG_AB, hist_vol_masked, mri_vol_masked, image_size, scaling, factor):
    ### get indicies of the first and last slice of t2
    start = 0
    end = 0
    for z in range(mri_vol_masked.shape[0]):
        if np.sum(mri_vol_masked[z,:,:])> 100:
            end = z
    
    for z in reversed(range(mri_vol_masked.shape[0])):
        if np.sum(mri_vol_masked[z,:,:])> 100:
            start = z
    
    D, w_, h_, C = hist_vol_masked.shape
    
    hist_input = np.zeros((D,image_size,image_size,3))
    hist_input = hist_input.astype('float32')
    
    m = tf.constant([1.0,0,0,0,1.0,0])
    identity = np.zeros((D,6))
    for k in range(0,D):
        identity[k,:] = m
    
    #### get histology voluje
    for s in range(D):
    
        hist_pad_size = np.maximum(w_,h_)*2
        hist_pad = np.zeros((hist_pad_size, hist_pad_size, 3))
        x_offset = int((hist_pad_size - w_)/2)
        y_offset = int((hist_pad_size - h_)/2)

        hist_pad[x_offset:x_offset+w_,y_offset:y_offset+h_,:] = hist_vol_masked[s]
        hist_pad = hist_pad*255.0/np.max(hist_pad)
        hist_pad = hist_pad.astype('float32')
    
        points = np.argwhere( hist_pad[:,:,0] > 0)
        points = np.fliplr(points)
        y, x, h, w = cv2.boundingRect(points)
        
        if w>=h:
            x_offset = int(w*factor)
            y_offset = int((w - h + 2*x_offset)/2)
        else:
            y_offset = int(h*factor)
            x_offset = int((h - w + 2*y_offset)/2)
        
        hist_pad_crop = hist_pad[x - x_offset:x+w+x_offset, y - y_offset:y+h+y_offset,:]
        hist_pad_crop = cv2.resize(hist_pad_crop,(image_size,image_size))
        hist_input[s,:,:,:] = hist_pad_crop
        
    ### get MRI
    loss = np.zeros(end - D + 2 - start)

    hist_input_real = hist_input
    hist_input_torch = hist_input/255.0
    hist_input_torch = (hist_input_torch - 0.5)/0.5
    hist_input_torch = torch.from_numpy(np.transpose(hist_input_torch, (0, 3,1,2)))
    hist_input_torch = np.transpose(netG_AB.forward(Variable(hist_input_torch)).data.numpy(), (0, 2,3,1))
    hist_input = 255.0*(hist_input_torch*0.5 + 0.5)*(hist_input>0)
    
    for s in range(start, end - D + 2):
        mri_input = np.zeros((D,image_size,image_size,3))
        mri_input = mri_input.astype('float32')
        
        # = np.zeros((D,4*img_size,4*img_size,3))
        #mri_input_high_res = mri_input_high_res.astype('float32')
        for i in range(s,s+D):
            
            mri_vol_slice_gray = mri_vol_masked[i,:,:]
            
            mri_vol_slice = np.zeros((mri_vol_slice_gray.shape[0],mri_vol_slice_gray.shape[1],3))
            mri_vol_slice[:,:,0] = mri_vol_slice_gray
            mri_vol_slice[:,:,1] = mri_vol_slice_gray
            mri_vol_slice[:,:,2] = mri_vol_slice_gray
            
            points = np.argwhere( mri_vol_slice_gray > 0)
            points = np.fliplr(points)
            y_m, x_m, h_m, w_m = cv2.boundingRect(points)

            if w_m>=h_m:
                x_offset_m = int(w_m*factor)
                y_offset_m = int((w_m - h_m + 2*x_offset_m)/2)
            else:
                y_offset_m = int(h_m*factor)
                x_offset_m = int((h_m - w_m + 2*y_offset_m)/2)

            mri_vol_crop = mri_vol_slice[x_m - x_offset_m:x_m+w_m+x_offset_m, y_m - y_offset_m:y_m+h_m+y_offset_m,:]
            mri_vol_crop = cv2.resize(mri_vol_crop,(image_size,image_size))
            
            maxV = np.max(mri_vol_crop)
            minV = np.min(mri_vol_crop)
    
            mri_vol_crop = 255*(mri_vol_crop - minV)/(maxV - minV)

            #mri_vol_crop = mri_vol_crop*25.5/np.max(mri_vol_crop/10.0)
            
            mri_input[i-s,:,:,:] = mri_vol_crop

        # mri_mask_batch = tf.convert_to_tensor(np.where(mri_input > 0, 255.0, 0), dtype=tf.float32)
        # hist_mask_batch = tf.convert_to_tensor(np.where(hist_input > 0, 255.0, 0), dtype=tf.float32)
        # theta = model_aff(([mri_mask_batch, hist_mask_batch]))

        theta = model_aff(([mri_input,hist_input]))
        theta = tf.math.scalar_mul(scaling ,theta ) + identity
        
        hist_deformed = affine_transformer_network(hist_input,theta)
        
        
        theta_tps = model_tps(([mri_input,hist_deformed]))
        theta_tps = tf.math.scalar_mul(scaling, theta_tps )
        
        hist_vol_def = ThinPlateSpline(hist_deformed,theta_tps).numpy()
    
        ncc = NCC(mri_input, hist_vol_def)
        
        
        loss[s-start] = ncc
        
    return int(np.argmax(loss) + start)