# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:39:29 2019

@author: TMaysGGS
"""

'''Last updated on 11/12/2019 16:26'''
'''Importing the libraries'''
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, PReLU, Input, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dense, Dropout
from keras.utils import plot_model

from Tools.Keras_custom_layers import ArcFaceLossLayer 

NUM_LABELS = 67960

'''Building Block Functions'''
def conv_block(inputs, filters, kernel_size, strides, padding):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = PReLU(shared_axes = [1, 2])(Z)
    
    return A

def separable_conv_block(inputs, filters, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = SeparableConv2D(filters, kernel_size, strides = strides, padding = "same", use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = PReLU(shared_axes = [1, 2])(Z)
    
    return A

def bottleneck(inputs, filters, kernel, t, s, r = False):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    Z1 = conv_block(inputs, tchannel, 1, s, 'same')
    
    Z1 = DepthwiseConv2D(kernel, strides = 1, padding = "same", depth_multiplier = 1, use_bias = False)(Z1)
    Z1 = BatchNormalization(axis = channel_axis)(Z1)
    A1 = PReLU(shared_axes = [1, 2])(Z1)
    
    Z2 = Conv2D(filters, 1, strides = 1, padding = "same", use_bias = False)(A1)
    Z2 = BatchNormalization(axis = channel_axis)(Z2)
    
    if r:
        Z2 = add([Z2, inputs])
    
    return Z2

def inverted_residual_block(inputs, filters, kernel, t, strides, n):
    
    Z = bottleneck(inputs, filters, kernel, t, strides)
    
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel, t, 1, True)
    
    return Z

def linear_GD_conv_block(inputs, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = DepthwiseConv2D(kernel_size, strides = strides, padding = "valid", depth_multiplier = 1, use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    
    return Z

'''Building the MobileFaceNet Model'''
def mobile_face_net():
    
    X = Input(shape = (112, 112, 3))
    label = Input((NUM_LABELS, ))

    M = conv_block(X, 64, 3, 2, 'same') # Output Shape: (56, 56, 64) 

    M = separable_conv_block(M, 64, 3, 1) # (56, 56, 64) 
    
    M = inverted_residual_block(M, 64, 3, t = 2, strides = 2, n = 5) # (28, 28, 64) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 6) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1) # (7, 7, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 2) # (7, 7, 128) 
    
    M = conv_block(M, 512, 1, 1, 'valid') # (7, 7, 512) 
    
    M = linear_GD_conv_block(M, 7, 1) # (1, 1, 512) 
    # kernel_size = 7 for 112 x 112; 4 for 64 x 64
    
    M = conv_block(M, 128, 1, 1, 'valid')
    M = Dropout(rate = 0.1)(M)
    M = Flatten()(M)
    
    M = Dense(128, activation = None, use_bias = False, kernel_initializer = 'glorot_normal')(M) 
    
    Z_L = ArcFaceLossLayer(class_num = NUM_LABELS)([M, label])
    
    model = Model(inputs = [X, label], outputs = Z_L, name = 'mobile_face_net')
    
    return model

model = mobile_face_net()

model.summary()
model.layers

'''Loading the model & re-defining'''
model.load_weights('./Models/MobileFaceNet_train.h5')
# model.load_weights("E:\\Python_Coding\\MobileFaceNet\\model.hdf5")
model.summary()
model.layers

# Re-define the model
model.layers.pop() # Remove the ArcFace Loss Layer
model.layers.pop() # Remove the Label Input Layer
model.summary()

model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output] # Reset the output
output = model.get_layer(model.layers[-1].name).output
model.input
# The model used for prediction
pred_model = Model(model.input[0], output)
pred_model.summary()
pred_model.save('./Models/MobileFaceNet.h5')
plot_model(pred_model, to_file='pred_model.png')
