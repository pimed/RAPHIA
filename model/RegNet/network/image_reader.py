#### we referenced code from: https://github.com/ignacio-rocco/cnngeometric_pytorch

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from skimage import io
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
tf.keras.backend.set_floatx('float32')

### the output of each cost function is a tensor of shape TensorShape([batch_size])
def training_image_reader(csv_file, image_dir, output_shape = (224,224)):
    data = pd.read_csv(csv_file)
    mri_names = data.iloc[:,0]
    histology_fake_names = data.iloc[:,1]
    histology_names = data.iloc[:,2]
    mri_fake_names = data.iloc[:,3]
    
    num_of_images = len(mri_names)
    
    mri = np.zeros([num_of_images,output_shape[0],output_shape[1],3])
    histology = np.zeros([num_of_images,output_shape[0],output_shape[1],3])
    mri_fake = np.zeros([num_of_images,output_shape[0],output_shape[1],3])
    histology_fake = np.zeros([num_of_images,output_shape[0],output_shape[1],3])

    
    for idx in range(0,num_of_images):
        mri_image_path = os.path.join(image_dir,mri_names[idx])
        histology_image_path = os.path.join(image_dir,histology_names[idx])
        mri_fake_path = os.path.join(image_dir,mri_fake_names[idx])
        histology_fake_path = os.path.join(image_dir,histology_fake_names[idx])

    
        print(histology_image_path)
        mri_2d = cv2.imread(mri_image_path)
        mri_2d = cv2.resize(mri_2d, output_shape)
        
        
        histology_2d = cv2.imread(histology_image_path)
        histology_2d = cv2.resize(histology_2d, output_shape)
        
        
        mri_fake_2d = cv2.imread(mri_fake_path)
        mri_fake_2d = cv2.resize(mri_fake_2d, output_shape)
        histology_fake_2d = cv2.imread(histology_fake_path)
        histology_fake_2d = cv2.resize(histology_fake_2d, output_shape)
        

    
        mri[idx,:,:,:] = mri_2d
        histology[idx,:,:,:] = histology_2d
        mri_fake[idx,:,:,:] = mri_fake_2d
        histology_fake[idx,:,:,:] = histology_fake_2d

        mri = mri.astype('float32')
        histology = histology.astype('float32')
        mri_fake = mri_fake.astype('float32')
        histology_fake = histology_fake.astype('float32')

        
       
    dataset = {'mri': mri, 'histology': histology, 'mri_fake': mri_fake, 'histology_fake': histology_fake}
    
    return dataset
