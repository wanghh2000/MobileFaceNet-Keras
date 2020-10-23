# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:43:44 2020

@author: TMaysGGS
"""

"""Last updated on 2020.03.21 13:58"""
"""Importing the libraries"""
import os
import random
import tensorflow as tf
import numpy as np

"""Building helper functions"""
def _bytes_feature(value):
    
    '''Returns a bytes_list from a string / byte. '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList will not unpack a string from an EagerTensor. 
    
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    
    '''Returns a float_list from a float / double. '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _int64_feature(value):
    
    '''Returns an int64_list from bool / enum / int / uint. '''
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def convert_image_info_to_tfexample(anno):
    
    img_path = anno[0]
    label = int(anno[1])
    
    img_string = open(img_path, 'rb').read()
    img_shape = tf.image.decode_jpeg(img_string).shape
    
    feature = {
            'height': _int64_feature(img_shape[0]),
            'width': _int64_feature(img_shape[1]),
            'depth': _int64_feature(img_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(img_string),
            }
    
    return tf.train.Example(features = tf.train.Features(feature = feature))

"""Getting the image info"""
data_dir = r'/data/daiwei/processed_data/datasets_for_face_recognition'

label_list = os.listdir(data_dir)
img_info_list = []
for label in label_list:
    img_name_list = os.listdir(os.path.join(data_dir, label))
    for img_name in img_name_list:
        img_path = os.path.join(data_dir, label, img_name)
        img_info_list.append([img_path, label])
del label, img_name, img_path, label_list, img_name_list

random.shuffle(img_info_list)
img_num_per_tfrecord = 131072
img_info_sections = []
for i in range(int(np.ceil(len(img_info_list) / img_num_per_tfrecord))):
    temp_list = img_info_list[i * img_num_per_tfrecord: min((i + 1) * img_num_per_tfrecord, len(img_info_list))]
    img_info_sections.append(temp_list)

"""Writing the images & labels into TFRecord"""
tfrecord_save_prefix = r'/data/daiwei/processed_data/face_recognition_data_'

for i in range(len(img_info_sections)):
    tfrecord_save_path = tfrecord_save_prefix + str(i) + r'.tfrecord'
    with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
        for anno in img_info_sections[i]:
            tf_example = convert_image_info_to_tfexample(anno)
            writer.write(tf_example.SerializeToString())

# """Reading the images from TFRecord"""
# import tensorflow as tf
# import IPython.display as display
#
# tfrecord_save_path = r'F:\Datasets\data_1.tfrecord'
# raw_image_dataset = tf.data.TFRecordDataset(tfrecord_save_path)
#
# image_feature_description = {
#         'height': tf.io.FixedLenFeature([], tf.int64),
#         'width': tf.io.FixedLenFeature([], tf.int64),
#         'depth': tf.io.FixedLenFeature([], tf.int64),
#         'label': tf.io.FixedLenFeature([], tf.int64),
#         'image_raw': tf.io.FixedLenFeature([], tf.string),
#         }
#
# def _parse_image_function(example_proto):
#    
#     return tf.io.parse_single_example(example_proto, image_feature_description)
#
# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
#
# for image_feature in parsed_image_dataset:
#     image_raw = image_feature['image_raw'].numpy()
#     label = image_feature['label']
#     display.display(display.Image(data = image_raw))
#     print(label.numpy())
