# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:48:15 2021

 @author: Shenbin Qian

"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Conv2D, ReLU, LeakyReLU, Dropout, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model

def identity_block(X, k, filters):
    '''
    Create a two-layer ResNet identity block without BathNorm and a few ReLU
    '''
    # Retrieve Filters
    F1, F2 = filters
    # Save the input value which is needed later to add back to the main path. 
    X_shortcut = X
    # First component of main path
    X = Conv2DTranspose(filters = F1, kernel_size = (k, k), strides = (1,1), padding = 'same', activation='relu')(X)
  
    ## Second component of main path
    X = Conv2DTranspose(filters = F2, kernel_size = (k, k), strides = (1,1), padding = 'same')(X)
    
    ## Final step: Add shortcut value to main path
    X = Add()([X, X_shortcut])

    return X

def make_generator_model(input_shape=(35,26,1), factor=4):
    '''
    Define a Generator that use Functional model and ResNets (identity block)
    Varible -- factor: how many times that generator will increase the input LR
    Return: generator model
    '''
    h, w, C = input_shape # retrieve number of input channel
    k, s, n_f = 3, 1, 64  # set kernel size, stride and number of filters
    
    X_input = Input(input_shape)
    
    #add a fully connected layer to connect pixels
    X = Flatten()(X_input)  
    X = Dense(h*w*C, activation='relu')(X)
    X = Reshape((h, w, C))(X)

    #start using de-conv layers for learning features
    X = Conv2DTranspose(filters=n_f, kernel_size=(k,k), strides=(s,s), padding='same', activation='relu')(X)

    #for later use
    X_skip = X

    for i in range(16):
        X = identity_block(X, k, [n_f,n_f]) #16 ResNet identity block
      
    X = Conv2DTranspose(filters=n_f, kernel_size=(k,k), strides=(s,s), padding='same')(X)
    #add skip connection back
    X = Add()([X, X_skip])

    #Super resolution or  downscaling/upsampling layer
    n_F = (factor ** 2) * n_f
    X = Conv2DTranspose(filters=n_F, kernel_size=(k,k), strides=(s,s), padding='same')(X)
    #pixel shuffle -- rearranges data from depth into blocks of spatial data
    X = tf.nn.depth_to_space(input=X, block_size=factor) #output channel is n_F / block_size squared
    X = ReLU()(X)
 
    outputs = Conv2DTranspose(filters=C, kernel_size=(k,k), strides=(s,s), padding='same', activation='tanh')(X)  
    
    model = Model(inputs=X_input, outputs=outputs)
      
    return model

def make_discriminator_model(input_shape=(560,416,1)):
    '''
    Create a sequential model with 8 Conv2D layers with dropout rate 0.2
    '''
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))


    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(LeakyReLU())

    model.add(Dense(1, activation='sigmoid'))

    return model