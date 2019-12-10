# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:39:29 2019

@author: TMaysGGS
"""

'''Last updated on 12/10/2019 09:14'''
'''Importing the libraries'''
from keras.models import Model
from keras.utils import plot_model

from Model_Structures.MobileFaceNet import mobile_face_net_train 

NUM_LABELS = 67960 

'''Loading the model & re-defining''' 
model = mobile_face_net_train(NUM_LABELS) 
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
