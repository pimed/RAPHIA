import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tps import *

### compose two displacement field, (id+v)(id+u) - id
def Compose(theta_u,v1,v2):
    num_batch = v1.shape[0]
    size = v1.shape[1]
    axis_coords = np.linspace(-1,1,size)
    P_X, P_Y = np.meshgrid(axis_coords,axis_coords)
    id1 = np.zeros((num_batch,P_X.shape[0],P_X.shape[1]))
    id2 = np.zeros((num_batch,P_Y.shape[0],P_Y.shape[1]))
    for k in range(0,num_batch):
        id1[k,:,:] = P_X
    id1 = tf.convert_to_tensor(id1,dtype=tf.float32)
    for k in range(0,num_batch):
        id2[k,:,:] = P_Y
    id2 = tf.convert_to_tensor(id2,dtype=tf.float32)

    v1 = v1 + id1
    v2 = v2 + id2

    v1 = tf.expand_dims(v1,axis=3)
    v2 = tf.expand_dims(v2,axis=3)

    v1_def = ThinPlateSpline(v1, theta_u)
    v2_def = ThinPlateSpline(v2, theta_u)

    v1_def = tf.squeeze(v1_def,axis=3)
    v2_def = tf.squeeze(v2_def,axis=3)
    
    
    P_X[0,:] = 0
    P_X[:,0] = 0
    P_Y[0,:] = 0
    P_Y[:,0] = 0
    P_X[size-1,:] = 0
    P_X[:,size-1] = 0
    P_Y[size-1,:] = 0
    P_Y[:,size-1] = 0
    id1 = np.zeros((num_batch,P_X.shape[0],P_X.shape[1]))
    id2 = np.zeros((num_batch,P_Y.shape[0],P_Y.shape[1]))
    for k in range(0,num_batch):
        id1[k,:,:] = P_X
    id1 = tf.convert_to_tensor(id1,dtype=tf.float32)
    for k in range(0,num_batch):
        id2[k,:,:] = P_Y
    id2 = tf.convert_to_tensor(id2,dtype=tf.float32)

    v1_def = v1_def - id1
    v2_def = v2_def - id2
    
    return {"x_s" : v1_def, "y_s": v2_def}



### get displacement field from control vectors
def DispField(vector,size):


  def _repeat(x, n_repeats):
    rep = tf.transpose(
      tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  def _meshgrid(height, width, coord):
    x_t = tf.tile(
      tf.reshape(tf.linspace(-1.0, 1.0, width), [1, width]), [height, 1])
    y_t = tf.tile(
      tf.reshape(tf.linspace(-1.0, 1.0, height), [height, 1]), [1, width])

    x_t_flat = tf.reshape(x_t, (1, 1, -1))
    y_t_flat = tf.reshape(y_t, (1, 1, -1))

    num_batch = tf.shape(coord)[0]
    px = tf.expand_dims(coord[:,:,0], 2) # [bn, pn, 1]
    py = tf.expand_dims(coord[:,:,1], 2) # [bn, pn, 1]
    d2 = tf.square(x_t_flat - px) + tf.square(y_t_flat - py)
    r = d2 * tf.math.log(d2 + 1e-6) # [bn, pn, h*w]
    x_t_flat_g = tf.tile(x_t_flat, tf.stack([num_batch, 1, 1])) # [bn, 1, h*w]
    y_t_flat_g = tf.tile(y_t_flat, tf.stack([num_batch, 1, 1])) # [bn, 1, h*w]
    ones = tf.ones_like(x_t_flat_g) # [bn, 1, h*w]

    grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1) # [bn, 3+pn, h*w]
    
    return grid

  def _transform(T, coord, vector):
    num_batch = tf.shape(vector)[0]
    height = size
    width = size

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    out_height = size
    out_width = size
    grid = _meshgrid(out_height, out_width, coord) # [2, h*w]
    
    

    # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
    # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
    T_g = tf.matmul(T, grid) #
    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    
    axis_coords = np.linspace(-1,1,size)
    P_X, P_Y = np.meshgrid(axis_coords,axis_coords)
    offset_X = np.zeros((num_batch,P_X.shape[0],P_X.shape[1]))
    offset_Y = np.zeros((num_batch,P_Y.shape[0],P_Y.shape[1]))
    for k in range(0,num_batch):
        offset_X[k,:,:] = P_X
    offset_X = tf.convert_to_tensor(offset_X,dtype=tf.float32)
    for k in range(0,num_batch):
        offset_Y[k,:,:] = P_Y
    offset_Y = tf.convert_to_tensor(offset_Y,dtype=tf.float32)
    
    

    x_s = tf.reshape(
      x_s, 
      tf.stack([num_batch, out_height, out_width]))
      
    y_s = tf.reshape(
      y_s, 
      tf.stack([num_batch, out_height, out_width]))
      
    x_s = x_s - offset_X
    y_s = y_s - offset_Y

    
    return {"x_s" : x_s, "y_s": y_s}

  def _solve_system(coord,vector):
    num_batch  = tf.shape(vector)[0]
    num_point  = tf.shape(vector)[1]

    ones = tf.ones([num_batch, num_point, 1], dtype="float32")
    p = tf.concat([ones, coord], 2) # [bn, pn, 3]

    p_1 = tf.reshape(p, [num_batch, -1, 1, 3]) # [bn, pn, 1, 3]
    p_2 = tf.reshape(p, [num_batch, 1, -1, 3]) # [bn, 1, pn, 3]
    d2 = tf.reduce_sum(tf.square(p_1-p_2), 3) # [bn, pn, pn]
    r = d2 * tf.math.log(d2 + 1e-6) # [bn, pn, pn]
    
    
    #### add regularization to the transformation
    #identity_matrix = tf.eye(r.shape[1])
    #identity_matrix = np.repeat(identity_matrix[np.newaxis,:,:],r.shape[0],axis=0)
    #r = r + 10.0*identity_matrix
    ### add regularization to the transformation

    zeros = tf.zeros([num_batch, 3, 3], dtype="float32")
    W_0 = tf.concat([p, r], 2) # [bn, pn, 3+pn]
    W_1 = tf.concat([zeros, tf.transpose(p, [0, 2, 1])], 2) # [bn, 3, pn+3]
    W = tf.concat([W_0, W_1], 1) # [bn, pn+3, pn+3]
    W_inv = tf.linalg.inv(W) 

    tp = tf.pad(coord+vector, 
      [[0, 0], [0, 3], [0, 0]], "CONSTANT") # [bn, pn+3, 2]
    T = tf.matmul(W_inv, tp) # [bn, pn+3, 2]
    T = tf.transpose(T, [0, 2, 1]) # [bn, 2, pn+3]

    return T
  ### the following code is added by Wei Shao
  vector = tf.reshape(vector,[vector.shape[0],-1,2])
  vector = tf.cast(vector, tf.float32)

  num_batch  = tf.shape(vector)[0]
  num_point  = tf.shape(vector)[1]
  grid_size = int(np.sqrt(num_point))
  axis_coords = np.linspace(-1,1,grid_size)
  P_X, P_Y = np.meshgrid(axis_coords,axis_coords)
  P_Y = np.reshape(P_Y,(-1,1)).squeeze()
  P_X = np.reshape(P_X,(-1,1)).squeeze()
  coord = np.zeros((num_point,2))
  coord[:,1] = P_X
  coord[:,0] = P_Y
  coord = np.repeat(coord[np.newaxis,:,:],num_batch,axis=0)
  coord = tf.convert_to_tensor(coord, dtype=np.float32)
  ### 
  

  T = _solve_system(coord, vector)
  output = _transform(T, coord, vector)
  return output