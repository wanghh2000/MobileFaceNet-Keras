# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:58:15 2019

@author: TMaysGGS
"""

'''Last updated on 12/10/2019 09:36'''
'''Importing the libraries & setting the configurations'''
import os 
import sys 
import keras 
from keras.optimizers import Adam 

sys.path.append('../') 
from Model_Structures.MobileFaceNet import mobile_face_net_train 

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

BATCH_SIZE = 128
NUM_LABELS = 67960 
m = 15090270
DATA_SPLIT = 0.005 
TOTAL_EPOCHS = 1000 

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

'''Training the Model'''
# Train on multiple GPUs
# from keras.utils import multi_gpu_model
# model = multi_gpu_model(model, gpus = 2)

model = mobile_face_net_train(NUM_LABELS) 
model.summary() 
model.layers

model.compile(optimizer = Adam(lr = 0.001, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Save the model after every epoch
from keras.callbacks import ModelCheckpoint 
check_pointer = ModelCheckpoint(filepath = '../Models/MobileFaceNet_train.h5', verbose = 1, save_best_only = True)

# Interrupt the training when the validation loss is not decreasing
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10000)

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
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 200, min_lr = 0)

'''Importing the data & training the model''' 
hist = model.fit_generator(
        train_generator,
        steps_per_epoch = 30000, # (m * (1 - DATA_SPLIT)) // BATCH_SIZE
        epochs = TOTAL_EPOCHS,
        callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr],
        validation_data = validate_generator, 
        validation_steps = (m * DATA_SPLIT) // BATCH_SIZE, 
        workers = 4, 
        use_multiprocessing = True, 
        initial_epoch = 0)

print(hist.history)
