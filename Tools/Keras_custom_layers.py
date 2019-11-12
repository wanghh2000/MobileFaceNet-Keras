# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:18:31 2019

@author: TMaysGGS
"""

'''Last updated on 11/12/2019 14:42'''
import math 
import tensorflow as tf 
import keras.backend as K 
from keras import initializers, regularizers, constraints 
from keras.layers import Layer 
from keras.engine.base_layer import InputSpec 
from keras.utils.generic_utils import to_list 

# Corrected PReLU Layer (Class)
class PReLU_Layer(Layer): 
    
    def __init__(self, alpha_initializer = {'class_name': 'Constant', 'config': {'value': .25}}, 
                 alpha_regularizer = None, 
                 alpha_constraint = None, 
                 shared_axes = None, 
                 **kwargs):
        
        super(PReLU_Layer, self).__init__(**kwargs)
        self.supports_masking = True 
        self.alpha_initializer = initializers.get(alpha_initializer) 
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint) 
        if shared_axes is None: 
            self.shared_axes = None 
        else:
            self.shared_axes = to_list(shared_axes, allow_tuple = True) 
    
    def build(self, input_shape): 
        
        param_shape = [input_shape[-1]] 
        self.param_broadcast = [False] * len(param_shape) 
        if self.shared_axes is not None: 
            for i in self.shared_axes:
                param_shape[i - 1] = 1 
                self.param_broadcast[i -1] = True 
        self.alpha = self.add_weight(shape = param_shape, 
                                     name = 'alpha', 
                                     initializer = self.alpha_initializer, 
                                     regularizer = self.alpha_regularizer, 
                                     constraint  =self.alpha_constraint) 
        
        axes = {} 
        if self.shared_axes: 
            for i in range(1, len(input_shape)): 
                if i not in self.shared_axes: 
                    axes[i] = input_shape[i] 
        self.input_spec = InputSpec(ndim = len(input_shape), axes = axes) 
        self.build = True 
        
    def call(self, inputs, mask = None): 
        
        pos = K.relu(inputs) 
        if K.backend() == 'tensorflow': 
            neg = -self.alpha * K.relu(-inputs)
        else:
            raise Exception("Only support TensorFlow backend for now.") 
        
        return pos + neg 
    
    def get_config(self): 
        
        config = {
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(PReLU_Layer, self).get_config() 
        
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape): 
        
        return input_shape

# Arc Face Loss Layer (Class)
class ArcFaceLossLayer(Layer):
    '''
    Arguments:
        inputs: the input embedding vectors
        class_num: number of classes
        s: scaler value (default as 64)
        m: the margin value (default as 0.5)
    Returns:
        the final calculated outputs
    '''
    def __init__(self, class_num, s = 64., m = 0.5, **kwargs):
        
        self.init = initializers.get('glorot_uniform') # Xavier uniform intializer
        self.class_num = class_num
        self.s = s
        self.m = m
        super(ArcFaceLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight((input_shape[0][-1], self.class_num), initializer = self.init, name = '{}_W'.format(self.name))
        super(ArcFaceLossLayer, self).build(input_shape)
        
    def call(self, inputs, mask = None):
        
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)
        
        # features
        X = inputs[0] 
        # 1-D or one-hot label works as mask
        Y_mask = inputs[1] 
        # If Y_mask is not in one-hot form, transfer it to one-hot form.
        if Y_mask.shape[-1] == 1: 
            Y_mask = K.cast(Y_mask, tf.int32)
            Y_mask = K.reshape(K.one_hot(Y_mask, self.class_num), (-1, self.class_num))
        
        X_normed = K.l2_normalize(X, axis = 1) # L2 Normalized X
        self.W = K.l2_normalize(self.W, axis = 0) # L2 Normalized Weights
        
        # cos(theta + m)
        cos_theta = K.dot(X_normed, self.W)
        cos_theta2 = K.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = K.sqrt(sin_theta2 + K.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))
        
        # This condition controls the theta + m should in range [0, pi]
        #   0 <= theta + m < = pi
        #   -m <= theta <= pi - m
        cond_v = cos_theta - threshold
        cond = K.cast(K.relu(cond_v), dtype = tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)
        
        # mask by label
        Y_mask =+ K.epsilon()
        inv_mask = 1. - Y_mask
        s_cos_theta = self.s * cos_theta
        
        output = K.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * Y_mask))
        
        return output
    
    def compute_output_shape(self, input_shape):
        
        return input_shape[0], self.class_num
