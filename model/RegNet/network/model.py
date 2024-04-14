import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

tf.keras.backend.set_floatx('float32')

def feature_extraction(input_shape, feature_cnn='vgg16'):
    x_in = Input(input_shape)
    if feature_cnn == 'vgg16' :
        model = tf.keras.Sequential()
        vgg16 = tf.keras.applications.VGG16(include_top=False, weights = 'imagenet', input_shape = input_shape)
            ### cropped at forth pooling layer, replace maximum pooling with global average pooling
       
        for i in range(0,10):
            vgg16.layers[i].trainable = False
            model.add(vgg16.layers[i])
        for i in range(10,14):
            model.add(vgg16.get_layer(index=i))
            model.add(tf.keras.layers.Dropout(0.25))
            #model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        
        y_out = model(x_in)


    if feature_cnn == 'densenet' :
        densenet = tf.keras.applications.densenet.DenseNet121(include_top=False,weights='imagenet', input_shape = input_shape)
        densenet_cut = Model(inputs=densenet.layers[0].input, outputs=densenet.layers[311].output)
        x_feature = densenet_cut(x_in)
        y_out = tf.keras.layers.GlobalAveragePooling2D()(x_feature)


    if feature_cnn == 'resnet' :
        resnet = tf.keras.applications.ResNet101(include_top=False,weights='imagenet', input_shape = input_shape)
        #resnet_cut = Model(inputs=resnet.layers[0].input, outputs=resnet.layers[80].output)
        resnet_cut = Model(inputs=resnet.layers[0].input, outputs=resnet.layers[125].output)
        x_feature = resnet_cut(x_in)
        y_out = tf.keras.layers.GlobalAveragePooling2D()(x_feature)
        
        
    return Model(inputs = x_in, outputs = y_out)

def feature_regression(input_shape = (512,), output_dim = 6):
    
    x_in = Input(input_shape)
    x = x_in
    fcl_model =  tf.keras.Sequential()
    fcl_model.add(Dense(output_dim))
    
    y_out = fcl_model(x_in)
    return Model(inputs = x_in, outputs = y_out)

def reg_net(mri_shape, hist_shape, feature_cnn='vgg16', transform_model='affine'):
    
    x_in_1 = Input(mri_shape)
    x_in_2 = Input(hist_shape)
    
    x_1 = x_in_1
    x_2 = x_in_2
    

    mri_feature_extraction = feature_extraction(input_shape = mri_shape, feature_cnn=feature_cnn)
    hist_feature_extraction = feature_extraction(input_shape = hist_shape, feature_cnn=feature_cnn)
     
    x_1 = mri_feature_extraction(x_1)
    x_2 = hist_feature_extraction(x_2)
    concatenate_feature = tf.concat([x_1,x_2],1)
    if transform_model == 'affine':
        regression = feature_regression(input_shape = (concatenate_feature.shape[1],), output_dim = 6)   
        y_out = regression(concatenate_feature)
    if transform_model == "tps":
        regression = feature_regression(input_shape = (concatenate_feature.shape[1],), output_dim = 72)   
        y_out = regression(concatenate_feature)

    return Model(inputs = [x_in_1,x_in_2], outputs = y_out)
        
    
