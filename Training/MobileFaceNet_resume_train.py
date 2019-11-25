# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:54:33 2019

@author: TMaysGGS
"""

'''Updated on 11/25/2019 09:52'''
'''Problem
There is an intermediate layer that has the shape (m, 128), which means a 26928 * 128 matrix. 
Each element takes 32 bit so that the total space needed is more than 21G, which way too much exceeds 
the GPU memory. 
'''
'''Loading the pre-trained model'''
import os
import sys 
import keras 
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, PReLU, Input, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dense, Dropout
from keras.optimizers import Adam
# import keras.backend.tensorflow_backend as KTF
# from keras.utils import plot_model

sys.path.append('../') 
from Tools.Keras_custom_layers import ArcFaceLossLayer 

os.environ['CUDA_VISIBLE_DEVICES'] = '3' # 如需多张卡设置为：'1, 2, 3'，使用CPU设置为：''
'''Set if the GPU memory needs to be restricted
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
session = tf.Session(config = config)

KTF.set_session(session)
'''
BATCH_SIZE = 128 
old_m = 15090270
m = 15090270 
DATA_SPLIT = 0.005
OLD_NUM_LABELS = 67960
NUM_LABELS = 67960 
TOTAL_EPOCHS = 1000 
IMG_DIR = '/data/daiwei/processed_data/datasets_for_face_recognition' 

'''Importing the data set'''
from keras.preprocessing.image import ImageDataGenerator

train_path = '/data/daiwei/processed_data/datasets_for_face_recognition'

train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = DATA_SPLIT)

def mobilefacenet_input_generator(generator, directory, subset):
    
    gen = generator.flow_from_directory(
            directory, 
            target_size = (112, 112), 
            color_mode = 'rgb', 
            batch_size = BATCH_SIZE, 
            class_mode = 'categorical', 
            subset = subset)
    
    while True: 
        
        X = gen.next()
        yield [X[0], X[1]], X[1] 

train_generator = mobilefacenet_input_generator(train_datagen, train_path, 'training')

validate_generator = mobilefacenet_input_generator(train_datagen, train_path, 'validation')

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
    
    M = Dense(128, activation = None, use_bias = False)(M) 

    Z_L = ArcFaceLossLayer(class_num = OLD_NUM_LABELS)([M, label])
    
    model = Model(inputs = [X, label], outputs = Z_L, name = 'mobile_face_net')
    
    return model

model = mobile_face_net()

# model.summary()
# model.layers

'''Loading the model & re-defining'''
print("Reading the pre-trained model") 
model.load_weights(r'../Models/MobileFaceNet_train.h5')
# model.load_weights("E:\\Python_Coding\\MobileFaceNet\\tl_model_1905270955.hdf5")
print("Reading done. ")
model.summary()
# model.layers

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
# pred_model.save('pred_model.h5')

# Custom the model for continue training
label = Input((NUM_LABELS, ))
M = pred_model.output
Z_L = ArcFaceLossLayer(class_num = NUM_LABELS)([M, label])

customed_model = Model(inputs = [pred_model.input, label], outputs = Z_L, name = 'mobile_face_net_transfered')
customed_model.summary()
# customed_model.layers
# plot_model(customed_model, to_file='customed_model.png')

'''Setting configurations for training the Model''' 
customed_model.compile(optimizer = Adam(lr = 0.01, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Temporarily increase the learing rate to 0.01

# Save the model after every epoch
from keras.callbacks import ModelCheckpoint 
check_pointer = ModelCheckpoint(filepath = '../Models/MobileFaceNet_train.h5', verbose = 1, save_best_only = True)

# Interrupt the training when the validation loss is not decreasing
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1000)

# Record the loss history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
        
    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# Stream each epoch results into a .csv file
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('training.csv', separator = ',', append = True)
# append = True append if file exists (useful for continuing training)
# append = False overwrite existing file

# Reduce learning rate when a metric has stopped improving
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 20, min_lr = 0)

'''Importing the data & training the model'''
hist = customed_model.fit_generator(
        train_generator,
        steps_per_epoch = 2000, # (m * (1 - DATA_SPLIT)) // BATCH_SIZE,
        epochs = TOTAL_EPOCHS,
        callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr], 
        validation_data = validate_generator, 
        validation_steps = (m * DATA_SPLIT) // BATCH_SIZE, 
        workers = 4, 
        use_multiprocessing = True, 
        initial_epoch = 9)

