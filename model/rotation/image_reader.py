import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from skimage import io
import pandas as pd
import os
tf.keras.backend.set_floatx('float32')

### the output of each cost function is a tensor of shape TensorShape([batch_size])
def image_reader(csv_file, image_dir, output_shape = (256,256)):
    data = pd.read_csv(csv_file)
    histology_names = data.iloc[:,0]
    ink_names = data.iloc[:,1]
    num_of_images = len(histology_names)
    histology = np.zeros([num_of_images,output_shape[0],output_shape[1],3])
    ink = np.zeros([num_of_images,output_shape[0],output_shape[1],3])
    
    for idx in range(0,num_of_images):
        histology_image_path = os.path.join(image_dir,histology_names[idx])
        print(histology_image_path)
        histology_2d = cv2.imread(histology_image_path)
        histology_2d = cv2.resize(histology_2d, output_shape)
        histology[idx,:,:,:] = histology_2d
        histology = histology.astype('float32')

        ink_image_path = os.path.join(image_dir,ink_names[idx])
        print(ink_image_path)
        ink_2d = cv2.imread(ink_image_path)
        ink_2d = cv2.resize(ink_2d, output_shape)
        ink[idx,:,:,:] = ink_2d
        ink = ink.astype('float32')
    
    return histology, ink
