# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:51:46 2021

@author: Shenbin Qian
"""

import tensorflow as tf
from networks import make_generator_model, make_discriminator_model
from trainer import Trainer
from utils import read_raw_data_for_train, generate_LR, normalise
import gc


if __name__ == '__main__':
    
    
    # restore checkpoint
    generator = make_generator_model()
    discriminator = make_discriminator_model(input_shape=(140, 104, 1))
    
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator)
    
    checkpoint.restore(tf.train.latest_checkpoint('./trained_models/temperature/training_checkpoints')).expect_partial()
    
    HR = read_raw_data_for_train(file_dir="../mnt", extract_data="temperature")
    HR_norm = normalise(generate_LR(HR, factor=4))
    LR_norm = normalise(generate_LR(HR, factor=16))
    del HR
    gc.collect()
    
    trainer = Trainer(HR_norm, LR_norm, generator, discriminator, EPOCHS=200)

    #trainer.trainCNN()
    trainer.train()