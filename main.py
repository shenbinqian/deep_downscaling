# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:51:46 2021

@author: Shenbin Qian
"""


from networks import make_generator_model, make_discriminator_model
from trainer import Trainer
from utils import read_raw_data_for_train, generate_LR, normalise
import gc


if __name__ == '__main__':
    
    '''
    # we can also restore checkpoint where trained models are saved
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator)
    
    checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints'))
    '''
    
    #read all temperature data
    HR = read_raw_data_for_train(file_dir="../mnt", extract_data="temperature")
    #generate LR data and normalise
    HR_norm = normalise(generate_LR(HR, factor=4)) #here HR_norm refers to nomalised MR data
    LR_norm = normalise(generate_LR(HR, factor=16))
    del HR #delete HR for saving RAM
    gc.collect()
    
    G = make_generator_model()
    D = make_discriminator_model(input_shape=(140, 104, 1))
    
    trainer = Trainer(HR_norm, LR_norm, G, D, EPOCHS=1000)

    trainer.trainCNN() #train CNN
    trainer.train(epochs_for_GAN=200) #train GAN