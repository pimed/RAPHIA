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

### import self-defined functions
from model import *
from image_reader import *
from loss import *
from transform import *
from combine_transforms import *


tf.keras.backend.set_floatx('float32')


def build_parser():
    parser = ArgumentParser()
    
    # Paths
    parser.add_argument('--image-dir', type=str, default='../preprocessing/Stanford/', help='path to foler of training images')
    parser.add_argument('--csv-file', type=str, default='../training_data/train.csv', help='path to csv file of training examples')
    parser.add_argument('--validation-csv-file', type=str, default='../training_data/validation.csv', help='path to csv file of training examples')
    parser.add_argument('--trained-model-dir', type=str, default='../trained_models/', help='path to trained models folder')
    parser.add_argument('--trained-model-fn', type=str, default='GcGAN_Pseudo_Multimodal_', help='trained model filename')
    parser.add_argument('--result-name', type=str, default='../trained_models/results.csv', help='directory to store registration results')
    # Optimization parameters 
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--initial-img-size', type=int, default=128, help='image size at first resolution')
    # Model parameters
    parser.add_argument('--feature-cnn', type=str, default='vgg16', help='Feature extraction network: vgg16/densenet/resnet')
    
    return parser
   
   
def main():
    num_of_points = 72
    size = 224
    
    parser = build_parser()
    args = parser.parse_args()
    
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_visible_devices(devices[args.gpu_id], 'GPU')
    
    factor = 0.0001
    scaling_affine = tf.constant([factor,factor,factor,factor,factor,factor])
    scaling_affine = tf.expand_dims(scaling_affine,axis=0)
    scaling_tile_affine = tf.tile(scaling_affine,[args.batch_size,1])
    
    scaling_tps = tf.fill([num_of_points], factor)
    scaling_tps = tf.expand_dims(scaling_tps,axis=0)
    scaling_tile_tps = tf.tile(scaling_tps,[args.batch_size,1])
    
    train_losses = np.zeros(args.num_epochs)
    validation_losses = np.zeros(args.num_epochs)
    
    m = tf.constant([1.0,0,0,0,1.0,0])
    
    data = pd.read_csv(args.csv_file)
    data_validation = pd.read_csv(args.validation_csv_file)
    
    num_of_train_images = data.shape[0]
    num_of_validation_images = data_validation.shape[0]
    

    
    best_loss = 0
    image_size = args.initial_img_size
        
    dataset = training_image_reader(args.csv_file, args.image_dir, output_shape = (image_size,image_size))
    dataset_validation = training_image_reader(args.validation_csv_file, args.image_dir, output_shape = (image_size,image_size))
        
    mri = dataset['mri']
    hist = dataset['histology']
    mri_fake = dataset['mri_fake']
    hist_fake = dataset['histology_fake']
        

    mri_shape = mri.shape[1:4]
    hist_shape = hist.shape[1:4]
        
    mri_validation = dataset_validation['mri']
    hist_validation = dataset_validation['histology']
    mri_fake_validation = dataset_validation['mri_fake']
    hist_fake_validation = dataset_validation['histology_fake']
         
        
    model_affine = reg_net(mri_shape, hist_shape, feature_cnn=args.feature_cnn, transform_model='affine')
    model_tps = reg_net(mri_shape, hist_shape, feature_cnn=args.feature_cnn, transform_model='tps')
    

    optimizer_affine = tf.keras.optimizers.Adam(learning_rate=args.lr)
    optimizer_tps = tf.keras.optimizers.Adam(learning_rate=args.lr)

  
    for epoch in range(1,args.num_epochs+1):
        num_of_images = mri.shape[0]
        num_of_batches = int(num_of_images/args.batch_size)
        s_affine = 0
        s_tps = 0
            
        for idx in range(0,num_of_batches):
            batch_idx = np.random.randint(num_of_images, size=args.batch_size)
            mri_batch = mri[batch_idx, :]
            hist_batch = hist[batch_idx, :]
            mri_fake_batch = mri_fake[batch_idx, :]
            hist_fake_batch = hist_fake[batch_idx, :]
            
            
            #### begin of randomly deform the MRI and histopathology images to create synthetic image pairs for training
            t_affine = np.zeros((args.batch_size,6))
            
            for i in range(0,args.batch_size):
                angle = np.random.uniform(-np.pi/10,np.pi/10)
                shear_x = 0
                shear_y = 0
                trans_x = 0 #np.random.uniform(-1,1)/20
                trans_y = 0 #np.random.uniform(-1,1)/20
                scale_x = np.random.uniform(0.8,1.2)
                scale_y = np.random.uniform(0.8,1.2)
                # computer matrix A
                shear_matrix = np.array([[1,shear_x],[shear_y,1]])
                rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                scaling_matrix = np.array([[scale_x,0],[0,scale_y]])
                A = (rotation_matrix.dot(scaling_matrix)).dot(shear_matrix)
                t_affine[i][0] = A[0][0]
                t_affine[i][1] = A[0][1]
                t_affine[i][2] = trans_x
                t_affine[i][3] = A[1][0]
                t_affine[i][4] = A[1][1]
                t_affine[i][5] = trans_y
                
            hist_fake_batch_affine = affine_transformer_network(hist_fake_batch, t_affine)
            
            
            t_affine = np.zeros((args.batch_size,6))
            
            for i in range(0,args.batch_size):
                angle = np.random.uniform(-np.pi/10,np.pi/10)
                shear_x = 0
                shear_y = 0
                trans_x = 0 # np.random.uniform(-1,1)/20
                trans_y = 0 #np.random.uniform(-1,1)/20
                scale_x = np.random.uniform(0.8,1.2)
                scale_y = np.random.uniform(0.8,1.2)
                # computer matrix A
                shear_matrix = np.array([[1,shear_x],[shear_y,1]])
                rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                scaling_matrix = np.array([[scale_x,0],[0,scale_y]])
                A = (rotation_matrix.dot(scaling_matrix)).dot(shear_matrix)
                t_affine[i][0] = A[0][0]
                t_affine[i][1] = A[0][1]
                t_affine[i][2] = trans_x
                t_affine[i][3] = A[1][0]
                t_affine[i][4] = A[1][1]
                t_affine[i][5] = trans_y

            hist_batch_affine = affine_transformer_network(hist_batch, t_affine)
            
            t_tps = np.zeros((args.batch_size,num_of_points))
            for i in range(0,args.batch_size):
                for j in range(0,args.batch_size):
                    t_tps[i][j] = np.random.uniform(-0.2,0.2)
                    
            hist_fake_batch_tps = ThinPlateSpline(hist_fake_batch,t_tps)
            
            t_tps = np.zeros((args.batch_size,num_of_points))
            for i in range(0,args.batch_size):
                for j in range(0,args.batch_size):
                    t_tps[i][j] = np.random.uniform(-0.2,0.2)
                    
            hist_batch_tps = ThinPlateSpline(hist_batch,t_tps)

            with tf.GradientTape(persistent=True) as tape:
                identity = np.zeros((args.batch_size,6))
                for k in range(0,args.batch_size):
                    identity[k,:] = m

                ### get mri and histology mask, add dice coefficient to loss function
                # use original data for training
                mri_mask_batch = tf.convert_to_tensor(np.where(mri_batch > 0, 255.0, 0), dtype=tf.float32)
                hist_mask_batch = tf.convert_to_tensor(np.where(hist_batch > 0, 255.0, 0), dtype=tf.float32)
                
                ### use intensity image as input
                #theta_multi_modal_affine = model_affine([mri_batch, hist_batch])
                ### use binary masks as input 
                theta_multi_modal_affine = model_affine([mri_mask_batch, hist_mask_batch])
                theta_multi_modal_affine = tf.math.multiply(scaling_tile_affine,theta_multi_modal_affine) + identity
                hist_mask_affinely_deformed = affine_transformer_network(hist_mask_batch, theta_multi_modal_affine)
                hist_affinely_deformed = affine_transformer_network(hist_batch, theta_multi_modal_affine)
                affine_dice_loss = dice_loss(mri_mask_batch, hist_mask_affinely_deformed)/args.batch_size

                theta_multi_modal_tps = model_tps([mri_batch, hist_affinely_deformed])
                #print(theta_multi_modal_tps.shape)
                #print(scaling_tile_tps.shape)
                theta_multi_modal_tps = tf.math.multiply(scaling_tile_tps ,theta_multi_modal_tps)
                hist_mask_tps_deformed = ThinPlateSpline(hist_mask_affinely_deformed,theta_multi_modal_tps)
                tps_dice_loss = dice_loss(mri_mask_batch, hist_mask_tps_deformed)/args.batch_size


                #### use fake data for training
                theta_affine_hist = model_affine([mri_fake_batch, hist_batch_affine])
                
                theta_affine_hist = tf.math.multiply(scaling_tile_affine ,theta_affine_hist) + identity
                hist_affine_deformed_batch = affine_transformer_network(hist_batch_affine, theta_affine_hist)
                ssd_affine_hist = MSE(hist_batch, hist_affine_deformed_batch)
                
                theta_affine_mri = model_affine([mri_batch, hist_fake_batch_affine])
                identity = np.zeros((args.batch_size,6))
                for k in range(0,args.batch_size):
                    identity[k,:] = m
                theta_affine_mri = tf.math.multiply(scaling_tile_affine ,theta_affine_mri) + identity
                hist_fake_affine_deformed_batch = affine_transformer_network(hist_fake_batch_affine, theta_affine_mri)
                ssd_affine_mri = MSE(hist_fake_batch, hist_fake_affine_deformed_batch)
                
                
                theta_tps_hist = model_tps([mri_fake_batch, hist_batch_tps])
                theta_tps_hist = tf.math.multiply(scaling_tile_tps ,theta_tps_hist)
                hist_tps_deformed_batch = ThinPlateSpline(hist_batch_tps,theta_tps_hist)
                ssd_tps_hist = MSE(hist_batch, hist_tps_deformed_batch)

                
                theta_tps_mri = model_tps([mri_batch, hist_fake_batch_tps])
                theta_tps_mri = tf.math.multiply(scaling_tile_tps ,theta_tps_mri)
                hist_fake_tps_deformed_batch = ThinPlateSpline(hist_fake_batch_tps,theta_tps_mri)
                ssd_tps_mri = MSE(hist_fake_batch, hist_fake_tps_deformed_batch)
                

                
                output_hist = DispField(theta_tps_hist,size)
                x_s_hist = output_hist["x_s"]
                y_s_hist = output_hist["y_s"]
                smooth_tps_hist = smoothness(x_s_hist,y_s_hist)
                
                output_mri = DispField(theta_tps_mri,size)
                x_s_mri = output_mri["x_s"]
                y_s_mri = output_mri["y_s"]
                smooth_tps_mri = smoothness(x_s_mri,y_s_mri)
                            
                
                # loss_affine = -1.0*affine_dice_loss
                # loss_tps = -1.0*tps_dice_loss

                # loss_affine = 0.001*(ssd_affine_mri + ssd_affine_hist) 
                # loss_tps = 0.001*(ssd_tps_hist + ssd_tps_mri)  + 0.05*(smooth_tps_hist + smooth_tps_mri) 

                #loss_affine = 0.001*(ssd_affine_mri + ssd_affine_hist) + 1 - 1.0*affine_dice_loss

                loss_affine = 0.001*(ssd_affine_mri + ssd_affine_hist) + 1 - 1.0*affine_dice_loss
                loss_tps = 0.001*(ssd_tps_hist + ssd_tps_mri)  + 0.05*(smooth_tps_hist + smooth_tps_mri) + 1 - 1.0*tps_dice_loss

                

            gradients_affine = tape.gradient(loss_affine, model_affine.trainable_variables)
            optimizer_affine.apply_gradients(zip(gradients_affine,model_affine.trainable_variables))
            
            gradients_tps = tape.gradient(loss_tps, model_tps.trainable_variables)
            optimizer_tps.apply_gradients(zip(gradients_tps,model_tps.trainable_variables))
                
            ### sum up training loss
            s_affine = s_affine + loss_affine.numpy()
            s_tps= s_tps + loss_tps.numpy()
            
        ### compute validationing loss
        scaling_affine_val = tf.constant([factor,factor,factor,factor,factor,factor])
        scaling_affine_val = tf.expand_dims(scaling_affine_val,axis=0)
        identity_val = np.zeros((1,6))
        for k in range(0,1):
            identity_val[k,:] = m
            
        loss_validation_affine = 0
        loss_validation_tps = 0
        for idx in range(num_of_validation_images):
            mri_batch = np.expand_dims(mri_validation[idx],axis = 0)
            hist_batch = np.expand_dims(hist_validation[idx], axis = 0)

            mri_mask_batch = tf.convert_to_tensor(np.where(mri_batch > 0, 255.0, 0), dtype=tf.float32)
            hist_mask_batch = tf.convert_to_tensor(np.where(hist_batch > 0, 255.0, 0), dtype=tf.float32)
                
            # theta_aff = model_affine(([mri_batch,hist_batch]))
            theta_aff = model_affine([mri_mask_batch, hist_mask_batch])
            theta_aff = tf.math.scalar_mul(factor ,theta_aff ) + identity_val
            
            hist_batch_affine = affine_transformer_network(hist_batch,theta_aff)
            
            
            theta_tps = model_tps(([mri_batch,hist_batch_affine]))
            theta_tps = tf.math.scalar_mul(factor, theta_tps )
            hist_batch_tps = ThinPlateSpline(hist_batch_affine,theta_tps).numpy()
            
            hist_batch_affine_array = hist_batch_affine.numpy()

            
            loss_validation_affine = loss_validation_affine + dice_loss((np.where(mri_batch > 0, 255.0, 0)).astype(np.float32), (np.where(hist_batch_affine_array > 0, 255.0, 0)).astype(np.float32))
            loss_validation_tps = loss_validation_tps + dice_loss((np.where(mri_batch > 0, 255.0, 0)).astype(np.float32), (np.where(hist_batch_tps > 0, 255.0, 0)).astype(np.float32))

        
        loss_validation_affine = loss_validation_affine/num_of_validation_images
        loss_validation_tps = loss_validation_tps/num_of_validation_images
        
        if loss_validation_tps > best_loss:
            best_loss = loss_validation_tps
            model_affine.save(args.trained_model_dir + args.trained_model_fn + args.feature_cnn + '_affine_best_loss' + '.h5')
            model_tps.save(args.trained_model_dir + args.trained_model_fn + args.feature_cnn + '_tps_best_loss' + '.h5')
            
        if epoch%5 == 0:
            model_affine.save(args.trained_model_dir + args.trained_model_fn + args.feature_cnn + '_affine_epoch_' + str(epoch) + '.h5')
            model_tps.save(args.trained_model_dir + args.trained_model_fn + args.feature_cnn + '_tps_epoch_' + str(epoch) + '.h5')
            
        print("epoch= " + str(epoch) + ",  train affine loss = " + str(format(s_affine/num_of_batches, '.3f')) +
        ",  train tps loss = " + str(format(s_tps/num_of_batches, '.3f')) +
        ",  validation affine loss = " + str(format(loss_validation_affine, '.3f')) +
        ",  validation tps loss = " + str(format(loss_validation_tps, '.3f')))
            
        train_losses[epoch-1] = s_tps/num_of_batches
        validation_losses[epoch-1] = loss_validation_tps
            
        
    train_i = np.zeros((num_of_train_images,6))
    for k in range(0,num_of_train_images):
        train_i[k,:] = m
            
    validation_i = np.zeros((num_of_validation_images,6))
    for k in range(0,num_of_validation_images):
        validation_i[k,:] = m
        
        
    # save model for each image resolution
    model_affine.save(args.trained_model_dir + args.trained_model_fn + args.feature_cnn +  '_affine.h5')
    model_tps.save(args.trained_model_dir + args.trained_model_fn + args.feature_cnn +  '_tps.h5')
        
    
    
    print('done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
    np.savetxt(args.result_name.replace(".csv", args.feature_cnn + ".csv"), array, delimiter=",", fmt='%s')
    
    
if __name__ == "__main__":
    main()
