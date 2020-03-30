# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:39:29 2019

@author: TMaysGGS
"""

'''Last updated on 2020.03.29 03:14'''
'''Importing the libraries'''
from tensorflow.python.keras.models import Model

from Model_Structures.MobileFaceNet import mobile_face_net_train, mobile_face_net

NUM_LABELS = 67960
LOSS_TYPE = 'softmax'

'''Loading the training model'''
model = mobile_face_net_train(NUM_LABELS, loss = LOSS_TYPE)
model.load_weights('./Models/MobileFaceNet_train.h5')
model.summary()

pred_model = mobile_face_net()
pred_model.summary()

'''Extracting the weights & transfering to the prediction model'''
temp_weights_list = []
for layer in model.layers:
    
    if 'dropout' in layer.name:
        continue
    temp_layer = model.get_layer(layer.name)
    temp_weights = temp_layer.get_weights()
    temp_weights_list.append(temp_weights)

for i in range(len(pred_model.layers)):
    
    pred_model.get_layer(pred_model.layers[i].name).set_weights(temp_weights_list[i])
    
'''Verifying the results''' 
import numpy as np

x = np.random.rand(1, 112, 112, 3)
dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)
y1 = dense1_layer_model.predict(x)[0]
y2 = pred_model.predict(x)[0]
for i in range(128):
    assert y1[i] == y2[i]

'''Saving the model'''
pred_model.save(r'./Models/MobileFaceNet_tfkeras.h5')
