#### we referenced code from: https://github.com/voxelmorph/voxelmorph


import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
tf.keras.backend.set_floatx('float32')

from tps import *
from disp import *

def invCon(theta_u,theta_v,size):
    output_v = DispField(theta_v,size)
    v1 = output_v["x_s"]
    v2 = output_v["y_s"]
    
    def_v = Compose(theta_u,v1,v2)
    def_v1 = def_v["x_s"]
    def_v2 = def_v["y_s"]
    
    loss = tf.math.sqrt(tf.reduce_sum( def_v1*def_v1 + def_v2*def_v2)/(size*size*theta_u.shape[0]))*theta_u.shape[0]
    
    return loss
    


def smoothness(x_s, y_s):
    alpha = -0.75
    beta = -0.25
    gamma = 0.005

    dx_x,dx_y = tf.image.image_gradients(tf.expand_dims(x_s,axis=3))
    dy_x,dy_y = tf.image.image_gradients(tf.expand_dims(y_s,axis=3))

    dx_x_x, dx_x_y = tf.image.image_gradients(dx_x)
    dx_y_x, dx_y_y = tf.image.image_gradients(dx_y)
    dy_x_x, dy_x_y = tf.image.image_gradients(dy_x)
    dy_y_x, dy_y_y = tf.image.image_gradients(dy_y)
    
    L1 = tf.reduce_sum( (alpha*(dx_x_x + dx_y_y) + beta*(dx_x_x + dy_x_y ) + gamma* tf.expand_dims(x_s,axis=3))*(alpha*(dx_x_x + dx_y_y) + beta*(dx_x_x + dy_x_y ) + gamma* tf.expand_dims(x_s,axis=3)) )
    L2 = tf.reduce_sum( (alpha*(dy_x_x + dy_y_y) + beta*(dx_x_y + dy_y_y ) + gamma* tf.expand_dims(y_s,axis=3))*(alpha*(dy_x_x + dy_y_y) + beta*(dx_x_y + dy_y_y ) + gamma* tf.expand_dims(y_s,axis=3)) )
    
    loss = L1 + L2
    return loss
    
### the output of each cost function is a tensor of shape TensorShape([batch_size])
def losses(I,J,I_mask,J_mask,J_prime):
    #loss = -1.0*dice_loss(I_mask, J_mask) + 0.001*SSD(J, J_prime)
    #loss = SSD(J, J_prime)
    #loss = -ZNCC(J, J_prime)
    loss = - 1.0*ZNCC(J, J_prime)
    #loss = -1.0*dice_loss(I_mask, J_mask)
    return loss
    
def affine_losses(I,J,I_mask,J_mask):
    #loss = -1.0*dice_loss(I_mask, J_mask) + 0.001*SSD(J, J_prime)
    #loss = SSD(J, J_prime)
    #loss = -ZNCC(J, J_prime)
    #loss = -1.0*dice_loss(I_mask, J_mask) - 1.0*ZNCC(J, J_prime)
    loss = -1.0*dice_loss(I_mask, J_mask)
    return loss

### this function computes differences between adjacent slices in a 3D mask
### assume the size of the input is [batch_size,height, width, 3]
def spatialConsistency(mask):
    mask = mask[:,:,:,0]
    num_slices = mask.shape[0]
    loss = 0
    for i in range(0,num_slices - 1):
        loss = loss + tf.reduce_sum((mask[i,:,:] - mask[i+1,:,:] )*(mask[i,:,:] - mask[i+1,:,:]))/(255.0*255.0)
    loss = loss/num_slices
    return loss

def SSD(I,J):
    # assumes I, J are sized [batch_size, height, width,3]
    ssd = tf.reduce_sum(tf.math.multiply(I-J,I-J))
    return ssd/(I.shape[0]*I.shape[1]*I.shape[2]*I.shape[3]*255*255)
    
def MSE(I,J):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(I,J)

def dice_loss(I, J):
    I = I[:,:,:,0]
    J = J[:,:,:,0]

    # assumes I, J are sized [batch_size, height, width]
    numerator = 2 * tf.reduce_sum(I * J, [1,2])
    denominator = tf.maximum(tf.reduce_sum(I + J, [1,2]), 1e-5)
    dice = numerator/denominator
    return tf.reduce_sum(dice)/255.0


# zero_normalized cross-correlation (global)
def ZNCC(I, J):

    # first normalized both images to mean 0, Stdev 1 
    I_prime = normalize(I)
    J_prime = normalize(J)
    
    # compute CC squares
    IJ = tf.math.multiply(I_prime,J_prime)
    CC = tf.reduce_sum(IJ,axis=[1,2,3])/(I.shape[0]*I.shape[1]*I.shape[2]*I.shape[3])
    return tf.reduce_sum(CC)
    
def landmark_distance(I, J):  
    I = I[:,:,:,0]
    J = J[:,:,:,0]
    
    distance = np.zeros(I.shape[0])
    for i in range(0,I.shape[0]):
        ind_I = tf.where(tf.not_equal(I[i,:,:],0))
        center_I = tf.reduce_sum(ind_I,axis=0)/ind_I.shape[0]
        
        ind_J = tf.where(tf.not_equal(J[i,:,:],0))
        center_J = tf.reduce_sum(ind_J,axis=0)/ind_J.shape[0]
        
        distance[i] = tf.norm(center_I - center_J)
    
    return tf.reduce_sum(tf.convert_to_tensor(distance))

### normalize an image to mean 0, stdev 1 (global)
def normalize(I):
    num_of_pixels = I.shape[1]*I.shape[2]*I.shape[3]
    I_std = tf.math.reduce_std(I,axis=[1,2,3])
    I_mean = tf.math.reduce_sum(I,axis=[1,2,3])/num_of_pixels
    I_mean_expanded = np.zeros((I.shape[0],I.shape[1],I.shape[2],I.shape[3]))
    for i in range(0,I.shape[0]):
        I_mean_expanded[i,:,:,:] = I_mean[i]

    I_std_expanded = np.zeros((I.shape[0],I.shape[1],I.shape[2],I.shape[3]))
    for i in range(0,I.shape[0]):
        I_std_expanded[i,:,:,:] = I_std[i]
        
    I_prime = tf.divide(tf.math.subtract(I,I_mean_expanded),I_std_expanded)
    return I_prime
    
    
def NMI(I, J):
    MI(I, J)/MI(I, I)
    
    
def MI(y_true, y_pred,sigma_ratio=0.5):
    """
    Mutual Information for image-image pairs
        Building from neuron.losses.MutualInformationSegmentation()
    This function assumes that y_true and y_pred are both (batch_size x height x width x depth x nb_chanels)
    Author: Courtney Guo
    """


    """ prepare MI. """
    bin_centers = np.linspace(0,255,32)
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
    y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
    y_true = K.expand_dims(y_true, 2)
    y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
    y_pred = K.expand_dims(y_pred, 2)
    
    nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

    # reshape bin centers to be (1, 1, B)
    o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
    vbc = K.reshape(vol_bin_centers, o)
    
    # compute image terms
    I_a = K.exp(- preterm * K.square(y_true  - vbc))
    I_a /= K.sum(I_a, -1, keepdims=True)

    I_b = K.exp(- preterm * K.square(y_pred  - vbc))
    I_b /= K.sum(I_b, -1, keepdims=True)

    # compute probabilities
    I_a_permute = K.permute_dimensions(I_a, (0,2,1))
    pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
    pab /= nb_voxels
    pa = tf.math.reduce_mean(I_a, 1, keepdims=True)
    pb = tf.math.reduce_mean(I_b, 1, keepdims=True)
    
    papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
    mi = K.sum(K.sum(pab * tf.math.log(pab/papb + K.epsilon()), 1), 1)

    return mi
