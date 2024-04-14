from __future__ import print_function

import os
import sys
from argparse import ArgumentParser
from time import time

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import scipy.misc
from tensorflow.keras.optimizers import Adam

### import self-defined functions
from model import *
from image_reader import *
from loss import *
from transform import *


tf.keras.backend.set_floatx('float32')


def build_parser():
    parser = ArgumentParser()
    
    # Paths
    parser.add_argument('--image-dir', type=str, default='../preprocessing/Stanford/', help='path to foler of training images')
    parser.add_argument('--csv-file', type=str, default='../training_data/train.csv', help='path to csv file of training examples')
    parser.add_argument('--trained-model-dir', type=str, default='../trained_models/', help='path to trained models folder')
    parser.add_argument('--trained-model-fn', type=str, default='vgg16', help='trained model filename')
    parser.add_argument('--result-name', type=str, default='../trained_models/vgg16.csv', help='directory to store registration results')
    # Optimization parameters 
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=2500, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--initial-img-size', type=int, default=64, help='image size at first resolution')
    
    return parser
   
   
def main():

    parser = build_parser()
    args = parser.parse_args()
    
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_visible_devices(devices[args.gpu_id], 'GPU')
    
    train_losses = np.zeros(args.num_epochs)
    validation_losses = np.zeros(args.num_epochs)
    
    data = pd.read_csv(args.csv_file)


    factor = 1.0
    
    best_loss = 100000
    image_size = args.initial_img_size
        
    hist,ink = image_reader(args.csv_file, args.image_dir, output_shape = (image_size,image_size))

    val_ratio = 0.1

    num_of_images = data.shape[0]
    num_of_validation_images = int(val_ratio*num_of_images)
    num_of_train_images = num_of_images - num_of_validation_images

    train_idx = np.random.choice(range(num_of_images), num_of_train_images, replace=False).tolist()
    val_idx =  list(set(list(range(num_of_images))) - set(train_idx))

    hist_train = hist[train_idx,:,:,:]
    hist_validation = hist[val_idx,:,:,:]

    ink_train = ink[train_idx,:,:,:]
    ink_validation = ink[val_idx,:,:,:]

    print(hist_train.shape)
    print(hist_validation.shape)
    print(num_of_train_images)
    print(num_of_validation_images)


    hist_shape = hist_train.shape[1:4]
    ink_shape = ink_train.shape[1:4]
    model = angle_prediction(hist_shape, ink_shape)
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    for epoch in range(1,args.num_epochs+1):
        num_of_batches = int(num_of_train_images/args.batch_size)
        
        ssd_train = 0
            
        for idx in range(0,num_of_batches):
            #batch_idx = np.random.randint(num_of_train_images, size=args.batch_size)

            batch_idx = np.random.choice(range(num_of_train_images), args.batch_size, replace=False)

            hist_batch = hist_train[batch_idx, :, :, :]
            ink_batch = ink_train[batch_idx, :, :, :]
        
            train_angles = np.zeros((args.batch_size,1))
            train_flips = np.zeros((args.batch_size,1))
            t_affine = np.zeros((args.batch_size,6))
            for i in range(0,args.batch_size):
                angle = np.random.uniform(-factor*180,factor*180)
                train_angles[i] = angle
                train_flips[i] = np.random.randint(2)
                # computer matrix A
                A = np.array([[np.cos(angle*np.pi/180),-np.sin(angle*np.pi/180)],[np.sin(angle*np.pi/180),np.cos(angle*np.pi/180)]])
                t_affine[i][0] = A[0][0]
                t_affine[i][1] = A[0][1]
                t_affine[i][2] = 0
                t_affine[i][3] = A[1][0]
                t_affine[i][4] = A[1][1]
                t_affine[i][5] = 0
            
            for i in range(0,args.batch_size):
                if train_flips[i] == 1:
                    hist_batch[i] = cv2.flip(hist_batch[i] , 1)
                    ink_batch[i] = cv2.flip(ink_batch[i] , 1)
            
            hist_batch = affine_transformer_network(hist_batch, t_affine)
            ink_batch = affine_transformer_network(ink_batch, t_affine)
            
            
            ones = tf.ones(hist_batch.shape,tf.float32)
            indicies = tf.math.less(hist_batch, ones*10)
            indicies = tf.cast(indicies, tf.float32)
            hist_batch =  hist_batch + 244*indicies
            
            #cv2.imwrite('./rotated.png', hist_batch[0].numpy())
                
            train_angles = tf.convert_to_tensor(train_angles, dtype=tf.float32)
                
            with tf.GradientTape() as tape:
                    
                theta = model([hist_batch, ink_batch])

                loss = trig(theta*np.pi/180, train_angles*np.pi/180)



            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))

                
            ### sum up training loss
            ssd_train = ssd_train + loss.numpy()

            
        ### compute validationing loss
        
        validation_angles = np.zeros((num_of_validation_images,1))
        t_affine_val = np.zeros((num_of_validation_images,6))
        for i in range(0,num_of_validation_images):
            #angle = np.random.uniform(-np.pi*factor*epoch/args.num_epochs,np.pi*factor*epoch/args.num_epochs)
            angle = np.random.uniform(-180*factor,180*factor)
            validation_angles[i] = angle
            # computer matrix A
            A = np.array([[np.cos(angle*np.pi/180),-np.sin(angle*np.pi/180)],[np.sin(angle*np.pi/180),np.cos(angle*np.pi/180)]])
            t_affine_val[i][0] = A[0][0]
            t_affine_val[i][1] = A[0][1]
            t_affine_val[i][2] = 0
            t_affine_val[i][3] = A[1][0]
            t_affine_val[i][4] = A[1][1]
            t_affine_val[i][5] = 0
        
        hist_val_batch = affine_transformer_network(hist_validation, t_affine_val)
        ink_val_batch = affine_transformer_network(ink_validation, t_affine_val)
        
        ones = tf.ones(hist_val_batch.shape,tf.float32)
        indicies = tf.math.less(hist_val_batch, ones*10)
        indicies = tf.cast(indicies, tf.float32)
        hist_val_batch =  hist_val_batch + 244*indicies
        
        theta_validation = model([hist_val_batch, ink_validation])
        validation_angles = tf.convert_to_tensor(validation_angles, dtype=tf.float32)
        #if epoch/args.num_epochs < 0.1:
        #    loss_validation = SSD(theta_validation, validation_angles).numpy()
        #else:
        loss_validation = trig(theta_validation*np.pi/180, validation_angles*np.pi/180).numpy()

        
        if loss_validation < best_loss:
            best_loss = loss_validation
            model.save(args.trained_model_dir + args.trained_model_fn + '_angle_prediction_best_loss' + '.h5')
        
        print("epoch= " + str(epoch) + ",  train loss = " + str(format(ssd_train/num_of_batches, '.3f')) +
        ",  validation loss = " + str(format(loss_validation, '.3f')))
            
        train_losses[epoch-1] = ssd_train/num_of_batches
        validation_losses[epoch-1] = loss_validation
            
    print('done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    model.save(args.trained_model_dir + args.trained_model_fn + '_angle_prediction' + '.h5')
    array = np.empty((args.num_epochs + 10,4), dtype='U25')
    
    array[0,0] = "resolution"
    array[0,1] = "epoch"
    array[0,2] = "train_loss"
    array[0,3] = "validation_loss"
    
    for i in range(0,1):
        array[i*args.num_epochs + 1, 0] = str(i + 1)
        for j in range(0,args.num_epochs):
            array[i*args.num_epochs + 1 + j , 1] = str(j+1)
            array[i*args.num_epochs + 1 + j , 2] = str(train_losses[j])
            array[i*args.num_epochs + 1 + j , 3] = str(validation_losses[j])
    np.savetxt(args.result_name, array, delimiter=",", fmt='%s')
    
    
if __name__ == '__main__':
    main()
