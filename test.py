# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:32:49 2021

@author: Shenbin Qian
"""
from networks import make_generator_model, make_discriminator_model
from trainer import Trainer
import tensorflow as tf
from netCDF4 import Dataset
from utils import turn2train_shape, generate_LR, normalise

if __name__ == '__main__':
    
    # restore checkpoint where the trained model was saved
    generator = make_generator_model()
    discriminator = make_discriminator_model(input_shape=(140, 104, 1))
    
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator)
    
    checkpoint.restore(tf.train.latest_checkpoint('./trained_models/temperature/training_checkpoints')).expect_partial()

    #original test data, better different from training data
    atmos_data = Dataset('msc_20200115_metdata.nc')
    #choose a slice of the data set
    raw = tf.convert_to_tensor(atmos_data["air_pressure"][:560,:416,:70, 0].reshape(560, 416, 70, 1)) / 100
    train = turn2train_shape(raw) #turn to traiing shape
    
    #generate LR data and normalise
    HR_norm = normalise(generate_LR(train, factor=4))
    LR_norm = normalise(generate_LR(train, factor=16))
    
    trainer = Trainer(HR_norm, LR_norm, generator, discriminator)
    trainer.test_visualize_save_data(generator, 50, LR_norm) #default epochs is 50
