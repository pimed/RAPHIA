import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

tf.keras.backend.set_floatx('float32')


def feature_extraction(input_shape):
    model = tf.keras.Sequential()
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights = 'imagenet', input_shape = input_shape)
    x_in = Input(input_shape)
   
    for i in range(0,12):
        #vgg16.layers[i].trainable = False
        model.add(vgg16.layers[i])
    for i in range(12,14):
        model.add(vgg16.get_layer(index=i))
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # model = tf.keras.Sequential()
    # vgg16 = tf.keras.applications.VGG16(include_top=False, weights = 'imagenet', input_shape = input_shape)
    # x_in = Input(input_shape)
    # for i in range(0,15):
    #     vgg16.layers[i].trainable = False
    #     model.add(vgg16.layers[i])
    # for i in range(16,18):
    #     model.add(vgg16.get_layer(index=i))
    #     model.add(tf.keras.layers.Dropout(0.75))
        
    # model.add(vgg16.get_layer(index=18))
    # model.add(tf.keras.layers.Flatten())

    y_out = model(x_in)
    return Model(inputs = x_in, outputs = y_out)


def feature_regression(input_shape =  (512,)):
    x_in = Input(input_shape)
    x = x_in
    fcl_model =  tf.keras.Sequential()
    fcl_model.add(Dense(1))
    y_out = fcl_model(x_in)
    return Model(inputs = x_in, outputs = y_out)

def angle_prediction(hist_shape, ink_shape):
    x_in_1 = Input(hist_shape)
    x_in_2 = Input(ink_shape)
    x_1 = x_in_1
    x_2 = x_in_2

    ink_feature_extraction = feature_extraction(input_shape = ink_shape)
    hist_feature_extraction = feature_extraction(input_shape = hist_shape)
    # image_feature_extraction = feature_extraction(input_shape = hist_shape)
     
    x_1 = hist_feature_extraction(x_1)
    x_2 = ink_feature_extraction(x_2)

    # x_1 = image_feature_extraction(x_1)
    # x_2 = image_feature_extraction(x_2)


    concatenate_feature = tf.concat([x_1,x_2],1)

    regression = feature_regression(input_shape = (concatenate_feature.shape[1],))   
    y_out = regression(concatenate_feature)

    return Model(inputs = [x_in_1,x_in_2], outputs = y_out)
