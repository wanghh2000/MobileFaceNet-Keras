# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:54:33 2019

@author: TMaysGGS
"""

'''Last updated on 2020.03.26 23:42''' 
'''Importing the libraries & setting the configurations'''
import os
import sys
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

# import keras.backend.tensorflow_backend as KTF

sys.path.append('../')
from Model_Structures.MobileFaceNet import mobile_face_net_train
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
FC_LAYER_CHANGE = False 
OLD_LOSS_TYPE = 'softmax' 
LOSS_TYPE = 'arcface' 

'''Importing the data set'''
train_path = '/data/daiwei/processed_data/datasets_for_face_recognition'

train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = DATA_SPLIT)

def mobilefacenet_input_generator(generator, directory, subset, loss = 'arcface'):
    
    gen = generator.flow_from_directory(
            directory, 
            target_size = (112, 112), 
            color_mode = 'rgb', 
            batch_size = BATCH_SIZE, 
            class_mode = 'categorical', 
            subset = subset)
    
    while True: 
        
        X = gen.next() 
        if loss == 'arcface':
            yield [X[0], X[1]], X[1] 
        else: 
            yield X[0], X[1] 

train_generator = mobilefacenet_input_generator(train_datagen, train_path, 'training', LOSS_TYPE) 
validate_generator = mobilefacenet_input_generator(train_datagen, train_path, 'validation', LOSS_TYPE) 

'''Loading the model & re-defining''' 
model = mobile_face_net_train(OLD_NUM_LABELS, loss = OLD_LOSS_TYPE) 
print("Reading the pre-trained model... ") 
model.load_weights(r'../Models/MobileFaceNet_train.h5') 
print("Reading done. ") 
model.summary() 
# plot_model(model, to_file = r'../Models/training_model.png', show_shapes = True, show_layer_names = True) 
# model.layers

if OLD_NUM_LABELS == NUM_LABELS and not FC_LAYER_CHANGE and OLD_LOSS_TYPE == LOSS_TYPE: 
    customed_model = model 
    customed_model.summary() 
    
elif OLD_NUM_LABELS != NUM_LABELS or FC_LAYER_CHANGE or OLD_LOSS_TYPE != LOSS_TYPE: 
    # Re-define the model
    model.layers.pop() # Remove the ArcFace Loss Layer 
    model.layers.pop() # Remove the Label Input Layer 
    model.summary() 
    
    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output] # Reset the output
    output = model.get_layer(model.layers[-1].name).output
    
    # The model used for prediction 
    model.input 
    pred_model = Model(model.input[0], output) 
    pred_model.summary() 
    plot_model(pred_model, to_file = r'../Models/pred_model.png', show_shapes = True, show_layer_names = True) 
    # pred_model.save('pred_model.h5')

    # Custom the model for continue training
    if LOSS_TYPE == 'arcface': 
        label = Input((NUM_LABELS, ))
        M = pred_model.output
        Y = ArcFaceLossLayer(class_num = NUM_LABELS)([M, label]) 
        customed_model = Model(inputs = [pred_model.input, label], outputs = Y, name = 'mobile_face_net_transfered')
        customed_model.summary() 
    else: 
        M = pred_model.output 
        Y = Dense(units = NUM_LABELS, activation = 'softmax')(M) 
        customed_model = Model(inputs = pred_model.input, outputs = Y, name = 'mobile_face_net_transfered') 
    # customed_model.layers
    # plot_model(customed_model, to_file='customed_model.png') 

# Use multi-gpus to train the model 
num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) 
if num_gpus > 1:
    parallel_model = multi_gpu_model(customed_model, gpus = num_gpus) 

'''Setting configurations for training the Model''' 
if num_gpus == 1:
    customed_model.compile(optimizer = Adam(lr = 0.01, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Temporarily increase the learing rate to 0.01
else:
    parallel_model.compile(optimizer = Adam(lr = 0.01, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Save the model after every epoch
class ParallelModelCheckpoint(ModelCheckpoint): 
    
    def __init__(self, model, filepath, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1): 
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
    
    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)

if num_gpus == 1:
    check_pointer = ModelCheckpoint(filepath = '../Models/MobileFaceNet_train.h5', verbose = 1, save_best_only = True)
elif num_gpus > 1:
    check_pointer = ParallelModelCheckpoint(customed_model, filepath = '../Models/MobileFaceNet_train.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

# Interrupt the training when the validation loss is not decreasing
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1000)

# Record the loss history
class LossHistory(Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
        
    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# Stream each epoch results into a .csv file
csv_logger = CSVLogger('training.csv', separator = ',', append = True)
# append = True append if file exists (useful for continuing training)
# append = False overwrite existing file

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 20, min_lr = 0) 

'''Importing the data & training the model'''
# Model.fit_generator is deprecated and will be removed in a future version, 
# Please use Model.fit, which supports generators.
if num_gpus == 1:
    hist = customed_model.fit(
            train_generator,
            steps_per_epoch = int(m * (1 - DATA_SPLIT) / BATCH_SIZE), 
            epochs = TOTAL_EPOCHS,
            callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr], 
            validation_data = validate_generator, 
            validation_steps = int(m * DATA_SPLIT / BATCH_SIZE), 
            workers = 1, 
            use_multiprocessing = False, 
            initial_epoch = 3)
elif num_gpus > 1:
    hist = parallel_model.fit(
            train_generator,
            steps_per_epoch = int(m * (1 - DATA_SPLIT) / BATCH_SIZE), 
            epochs = TOTAL_EPOCHS,
            callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr], 
            validation_data = validate_generator, 
            validation_steps = int(m * DATA_SPLIT / BATCH_SIZE), 
            workers = 1, 
            use_multiprocessing = False, 
            initial_epoch = 3)
