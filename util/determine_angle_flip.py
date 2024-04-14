# reference to https://github.com/CODAIT/deep-histopath/blob/master/deephistopath/wsi/filter.py

import glob
import numpy as np
from PIL import Image
import scipy.ndimage.morphology as sc_morph
import skimage.morphology as sk_morph
from scipy import ndimage as ndi
import argparse
import cv2
from transform import *
import torch
from torch.autograd import Variable

def rotate_image(img, angles):
    t_affine = np.zeros((img.shape[0],6))
    for i in range(0,img.shape[0]):
        angle = angles[i]
        trans_x = 0
        trans_y = 0
        # computer matrix A
        A = np.array([[np.cos(angle*np.pi/180),-np.sin(angle*np.pi/180)],[np.sin(angle*np.pi/180),np.cos(angle*np.pi/180)]])
        t_affine[i][0] = A[0][0]
        t_affine[i][1] = A[0][1]
        t_affine[i][2] = trans_x
        t_affine[i][3] = A[1][0]
        t_affine[i][4] = A[1][1]
        t_affine[i][5] = trans_y
            
    hist_batch = affine_transformer_network(img, t_affine)

    ones = tf.ones(hist_batch.shape,tf.float32)
    indicies = tf.math.less(hist_batch, ones*10)
    indicies = tf.cast(indicies, tf.float32)
    hist_batch =  hist_batch + 244*indicies
    
    return hist_batch


def rotate_ink(ink, angles):
    t_affine = np.zeros((ink.shape[0],6))
    for i in range(0,ink.shape[0]):
        angle = angles[i]
        trans_x = 0
        trans_y = 0
        # computer matrix A
        A = np.array([[np.cos(angle*np.pi/180),-np.sin(angle*np.pi/180)],[np.sin(angle*np.pi/180),np.cos(angle*np.pi/180)]])
        t_affine[i][0] = A[0][0]
        t_affine[i][1] = A[0][1]
        t_affine[i][2] = trans_x
        t_affine[i][3] = A[1][0]
        t_affine[i][4] = A[1][1]
        t_affine[i][5] = trans_y
            
    ink_batch = affine_transformer_network(ink, t_affine)
    
    return ink_batch

def flip(ink):
    ink_blue = ink[:,:,0]
    if ndi.measurements.center_of_mass(ink_blue)[1] > ink_blue.shape[1]/2:
        flip = 0
    else:
        flip = 1
    return flip


def angle_flip_pred_AI_ink(model_angle, model_ink, img_fr, img_size = 64, ink_size = 1024, pad_ratio = 1.07):

    w, h, D = img_fr.shape
    img_fr =  cv2.resize(img_fr, (h,w))
    hist_sum = np.sum(img_fr, axis=-1)
    img_fr[:,:,0][hist_sum<100] = 244
    img_fr[:,:,1][hist_sum<100] = 244
    img_fr[:,:,2][hist_sum<100] = 244


    pad_size =  int(pad_ratio*np.maximum(w,h))
    x_offset = int((pad_size - w)/2)
    y_offset = int((pad_size - h)/2)

    img_pad = np.ones((pad_size, pad_size, 3))*244
    img_pad[img_pad.astype(int)==255] = 244
    img_pad = 255.0*img_pad/np.max(img_pad)
    img_pad[x_offset:x_offset+w,y_offset:y_offset+h,:] = cv2.cvtColor(img_fr, cv2.COLOR_BGR2RGB)

    hist_input_torch = np.float32(cv2.resize(img_pad , (ink_size, ink_size))/255.0)
    #hist_input_torch = np.float32(img_pad)/255.0

    hist_input_torch = np.expand_dims(hist_input_torch, axis = 0)
    hist_input_torch = (hist_input_torch - 0.5)/0.5
    hist_input_torch = torch.from_numpy(np.transpose(hist_input_torch, (0, 3,1,2)))
    ink_fr= np.transpose(model_ink.forward(Variable(hist_input_torch)).data.numpy(), (0, 2,3,1))
    ink_fr = np.squeeze(255.0*(ink_fr*0.5 + 0.5))


    ### ink image dilation
    kernel = np.ones((5,5), np.uint8)  
    ink_fr = cv2.dilate(ink_fr, kernel, iterations=1)  
    ink_fr = cv2.resize(ink_fr, (1024,1024))


    img_pad = np.ones((pad_size, pad_size, 3))*244
    img_pad[img_pad.astype(int)==255] = 244
    img_pad = 255.0*img_pad/np.max(img_pad)
    img_pad[x_offset:x_offset+w,y_offset:y_offset+h,:] = img_fr


    img_input = cv2.resize(img_pad,(img_size,img_size))
    # hist_sum = np.sum(img_input, axis=-1)
    # img_input[:,:,0][hist_sum<100] = 244
    # img_input[:,:,1][hist_sum<100] = 244
    # img_input[:,:,2][hist_sum<100] = 244
    img_input = np.expand_dims(img_input,axis = 0)
    img_input = img_input.astype('float32')

    ink_input = cv2.resize(cv2.cvtColor(ink_fr , cv2.COLOR_BGR2RGB),(img_size,img_size))
    ink_input = np.expand_dims(ink_input,axis = 0)
    ink_input = ink_input.astype('float32')

    angle = model_angle([img_input, ink_input])

    ink_pad_tf = np.expand_dims(cv2.cvtColor(ink_fr , cv2.COLOR_BGR2RGB),axis = 0)
    ink_pad_tf = ink_pad_tf.astype('float32')

    ink_pad_def = rotate_ink(ink_pad_tf,-angle).numpy()
    ink_pad_def = np.squeeze(ink_pad_def)
    
    to_flip = flip(ink_pad_def)
        
    if to_flip == 0:
        angle = (1.0*angle[0,:].numpy())#%360
    else:
        angle = (-1.0*angle[0,:].numpy())#%360
    
    return angle, to_flip


def angle_flip_pred_AI_ink_low_to_high_res(model_angle_low, model_angle_high, model_ink, img_fr, img_size_low = 64, img_size_high = 224, ink_size = 1024, pad_ratio = 1.07):

    w, h, D = img_fr.shape
    img_fr =  cv2.resize(img_fr, (h,w))
    hist_sum = np.sum(img_fr, axis=-1)
    img_fr[:,:,0][hist_sum<100] = 244
    img_fr[:,:,1][hist_sum<100] = 244
    img_fr[:,:,2][hist_sum<100] = 244


    pad_size =  int(pad_ratio*np.maximum(w,h))
    x_offset = int((pad_size - w)/2)
    y_offset = int((pad_size - h)/2)

    img_pad = np.ones((pad_size, pad_size, 3))*244
    img_pad[img_pad.astype(int)==255] = 244
    img_pad = 255.0*img_pad/np.max(img_pad)
    img_pad[x_offset:x_offset+w,y_offset:y_offset+h,:] = cv2.cvtColor(img_fr, cv2.COLOR_BGR2RGB)

    hist_input_torch = np.float32(cv2.resize(img_pad , (ink_size, ink_size))/255.0)

    hist_input_torch = np.expand_dims(hist_input_torch, axis = 0)
    hist_input_torch = (hist_input_torch - 0.5)/0.5
    hist_input_torch = torch.from_numpy(np.transpose(hist_input_torch, (0, 3,1,2)))
    ink_fr= np.transpose(model_ink.forward(Variable(hist_input_torch)).data.numpy(), (0, 2,3,1))
    ink_fr = np.squeeze(255.0*(ink_fr*0.5 + 0.5))


    ### ink image dilation
    kernel = np.ones((5,5), np.uint8)  
    ink_fr = cv2.dilate(ink_fr, kernel, iterations=1)  
    ink_fr = cv2.resize(ink_fr, (1024,1024))


    img_pad = np.ones((pad_size, pad_size, 3))*244
    img_pad[img_pad.astype(int)==255] = 244
    img_pad = 255.0*img_pad/np.max(img_pad)
    img_pad[x_offset:x_offset+w,y_offset:y_offset+h,:] = img_fr


    img_input = cv2.resize(img_pad,(img_size_low,img_size_low))
    img_input = np.expand_dims(img_input,axis = 0)
    img_input = img_input.astype('float32')

    ink_input = cv2.resize(cv2.cvtColor(ink_fr , cv2.COLOR_BGR2RGB),(img_size_low,img_size_low))
    ink_input = np.expand_dims(ink_input,axis = 0)
    ink_input = ink_input.astype('float32')

    angle = model_angle_low([img_input, ink_input])


    img_input = np.expand_dims(img_pad, axis=0)
    img_input = img_input.astype('float32')
    img_input = np.squeeze(rotate_image(img_input, -angle).numpy())
    img_input = cv2.resize(img_input, (img_size_high, img_size_high) )
    img_input = np.expand_dims(img_input,axis = 0)
    img_input = img_input.astype('float32')


    ink_input = np.expand_dims(cv2.cvtColor(ink_fr , cv2.COLOR_BGR2RGB),axis = 0)
    ink_input = ink_input.astype('float32')
    ink_input = np.squeeze(rotate_ink(ink_input, -angle).numpy())

    ink_input = cv2.resize(ink_input,(img_size_high,img_size_high))
    ink_input = np.expand_dims(ink_input,axis = 0)
    ink_input = ink_input.astype('float32')

    angle_ad = model_angle_high([img_input, ink_input])




    ink_pad_tf = np.expand_dims(cv2.cvtColor(ink_fr , cv2.COLOR_BGR2RGB),axis = 0)
    ink_pad_tf = ink_pad_tf.astype('float32')

    ink_pad_def = rotate_ink(ink_pad_tf,-angle-angle_ad).numpy()
    ink_pad_def = np.squeeze(ink_pad_def)
    
    to_flip = flip(ink_pad_def) 
    if to_flip == 1:
        angle = (-1.0*angle[0,:].numpy())#%360
        angle_ad = (-1.0*angle_ad[0,:].numpy())
    
    return angle + angle_ad, to_flip