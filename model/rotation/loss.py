#### we referenced code from: https://github.com/voxelmorph/voxelmorph

import tensorflow as tf

def SSD(I,J):
    ssd = tf.reduce_sum(tf.math.multiply(I-J,I-J))
    return ssd/I.shape[0]


def trig(I,J):
    # assumes I, J are sized [batch_size, 1]
    # loss = tf.reduce_sum( 0.5*(1.0000001 - tf.math.cos(I-J))  )
    
    loss = tf.reduce_sum(tf.math.sqrt(tf.math.sqrt(0.5*(1.0000001 - tf.math.cos(I-J)))))
    
    #loss = tf.reduce_sum(tf.math.sqrt(0.5*(1.0000001 - tf.math.cos(I-J)))) (error = 28 degrees))
    
    return loss/I.shape[0]
