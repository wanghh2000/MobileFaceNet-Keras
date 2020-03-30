# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:05:54 2020

@author: TMaysGGS
"""

'''Last updated on 2020.03.30 16:42'''
'''Importing the libraries & setting the configurations'''
import os
import sys
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizer_v2.adam import Adam

sys.path.append('../')
from Model_Structures.MobileFaceNet import mobile_face_net_train

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

BATCH_SIZE = 128
NUM_LABELS = 153 # 40
m = 34363 # 8922
DATA_SPLIT = 0.01
TOTAL_EPOCHS = 30
LOSS_TYPE = 'arcface'
OPTIMIZER = Adam
LR = 0.001

'''Loading the model'''
model = mobile_face_net_train(NUM_LABELS, loss = LOSS_TYPE) # change the loss to 'arcface' for fine-tuning
if LOSS_TYPE == 'arcface':
    INPUT_NAME_LIST = []
    for model_input in model.inputs:
        INPUT_NAME_LIST.append(model_input.name[: -2])

'''Importing the TFRecord(s)'''
# Directory where the TFRecords files are
tfrecord_save_dir = r'../Data'

tfrecord_path_list = []
for file_name in os.listdir(tfrecord_save_dir):
    if file_name[-9: ] == '.tfrecord':
        tfrecord_path_list.append(os.path.join(tfrecord_save_dir, file_name))

image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

if LOSS_TYPE == 'arcface':
    def _read_tfrecord(serialized_example):
        
        example = tf.io.parse_single_example(serialized_example, image_feature_description)
        
        img = tf.image.decode_jpeg(example['image_raw'], channels = 3) # RGB rather than BGR!!! 
        # img = tf.cast(img, tf.uint8)
        img = tf.cast(img, tf.float32) / 255.
        img_shape = [example['height'], example['width'], example['depth']]
        img = tf.reshape(img, img_shape)
        
        label = example['label']
        one_hot_label = tf.one_hot(label, NUM_LABELS)
        
        return {INPUT_NAME_LIST[0]: img, INPUT_NAME_LIST[1]: one_hot_label}, one_hot_label
    
else:
    def _read_tfrecord(serialized_example):
        
        example = tf.io.parse_single_example(serialized_example, image_feature_description)
        
        img = tf.image.decode_jpeg(example['image_raw'], channels = 3) # RGB rather than BGR!!! 
        # img = tf.cast(img, tf.uint8)
        img = tf.cast(img, tf.float32) / 255.
        img_shape = [example['height'], example['width'], example['depth']]
        img = tf.reshape(img, img_shape)
        
        label = example['label']
        one_hot_label = tf.one_hot(label, NUM_LABELS)
        
        return img, one_hot_label

raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path_list)
parsed_image_dataset = raw_image_dataset.map(_read_tfrecord)
parsed_image_dataset = parsed_image_dataset.cache()
parsed_image_dataset = parsed_image_dataset.repeat()
parsed_image_dataset = parsed_image_dataset.shuffle(131072)
parsed_image_dataset = parsed_image_dataset.batch(BATCH_SIZE)

'''Training the Model'''
model.compile(optimizer = OPTIMIZER(lr = LR, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Save the model after every epoch
check_pointer = ModelCheckpoint(filepath = '../Models/MobileFaceNet_train.h5', verbose = 1, save_best_only = False)

# Interrupt the training when the validation loss is not decreasing
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10000)

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
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 200, min_lr = 0)

'''Importing the data & training the model'''
# Model.fit_generator is deprecated and will be removed in a future version, 
# Please use Model.fit, which supports generators.
hist = model.fit(
        parsed_image_dataset, 
        steps_per_epoch = int(m // BATCH_SIZE), 
        epochs = TOTAL_EPOCHS, 
        callbacks = [check_pointer, history, csv_logger])

print(hist.history)