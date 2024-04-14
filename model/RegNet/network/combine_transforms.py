import tensorflow as tf
import numpy as np

def combine_affine_transforms(theta_pre, theta_con):
    ### assume that both theta matricies are of size [batch_size, 6]
    B = theta_pre.shape[0]
    theta_pre = tf.reshape(theta_pre, [B,2,3])
    theta_con = tf.reshape(theta_con, [B,2,3])
    
    
    output_list = []
    
    for i in range(0,B):

        m = tf.linalg.matmul(theta_con[i,0:2,0:2],theta_pre[i,0:2,0:2])
        t = tf.linalg.matmul(theta_con[i,0:2,0:2],tf.reshape(theta_pre[i,0:2,2],[2,1])) + tf.reshape(theta_con[i,0:2,2],[2,1])
        
        output_list.append(tf.reshape(tf.concat([m,t],axis = 1),[6]))
        
    theta_composite = tf.stack(output_list)
    
    return theta_composite
    
