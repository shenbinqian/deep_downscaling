# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:48:15 2021

@author: Shenbin Qian
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import time, os
from IPython import display

class Trainer():
    '''
    Define a Trainer to deal with training steps, loss function, optimiser, 
    Data visualisation and checkpoint-saving
    '''
    def __init__(self, HR, LR, G, D, BATCH_SIZE=64, EPOCHS=50, alpha_advers=0.001, seeds=None):
        self.HR = HR
        self.LR = LR
        self.generator = G
        self.discriminator = D              
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.alpha_advers = alpha_advers #adverserial rate
        self.seeds = seeds #random samples of LR for showing results during training
        if self.seeds == None:
            self.seeds = self.LR[:5,:,:,:] #first 5 samples for default
        
    def get_batched(self):
        #turn dataset into expected batch size
        HR_batched = tf.data.Dataset.from_tensor_slices(self.HR).shuffle(60000).batch(self.BATCH_SIZE)
        LR_batched = tf.data.Dataset.from_tensor_slices(self.LR).shuffle(60000).batch(self.BATCH_SIZE)
        return HR_batched, LR_batched

    def plot_and_save_images(self, predictions, epoch, epochs_for_GAN):
        '''
        This fuction is to be called within 'test_visualize_save_data' for plotting images
        Variables: epoch, epochs_for_GAN are used to see if it's the last epoch during which we save our plots
        '''
        for ts in range(predictions.shape[0]):
            _, h, w, c = predictions.shape
            vmin, vmax = np.min(predictions[ts,:,:,:]), np.max(predictions[ts,:,:,:])
        
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            im = ax.imshow(tf.reshape(predictions[ts, :, :, :], (h,w,c)), vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
            plt.title('SR data at epoch{} example{}'.format(epoch, ts), fontsize=10)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            #save figure if it's the last epoch
            if epoch == self.EPOCHS or epoch == epochs_for_GAN:
                fig.savefig('genImage_epoch{}_example{}_{}.png'.format(epoch, ts, time.strftime('%m%d-%H%M%S')))  

        plt.show()

    def test_visualize_save_data(self, model, epoch, test_input, epochs_for_GAN=None):
        '''
        This function is used to generate data/images after generator is trained
        It can be called independently to see test results after training but epoch should be passed the same as EPOCHS
        Notice `training` is set to False, and model is the generator model
        When called during training, test_input is seeds
        Varibale: epochs_for_GAN might be used when training GANs with different epoch numbers
        '''
        predictions = model(test_input, training=False)

        if epoch == self.EPOCHS or epoch == epochs_for_GAN:
            #save generated SR data at the last epoch
            print('Generated SR data saved!')
            np.save('data_SR.npy', predictions)
        
        #show only 5 examples if test_input is larger than 5
        if predictions.shape[0] > 5:
            #randomly sample 5 examples from generated results
            id = np.random.randint(0,predictions.shape[0], 5)
            predictions = tf.gather(predictions, indices=id)

            self.plot_and_save_images(predictions, epoch, epochs_for_GAN)

        else:
            self.plot_and_save_images(predictions, epoch, epochs_for_GAN)

    def optimizer(self):
        #use Adam optimizer
        generator_optimizer = tf.keras.optimizers.Adam(1e-5)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
        return generator_optimizer, discriminator_optimizer

    def save_checkpoint(self):
        #instantiate a Checkpoint
        generator_optimizer, discriminator_optimizer = self.optimizer()
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        return checkpoint

    def loss_func(self, generated_SR, HR_batch, d_HR, d_SR):
        #calculate content loss first
        content_loss = tf.math.reduce_mean(input_tensor=tf.math.subtract(HR_batch, generated_SR)**2, axis=[1, 2, 3])

        #calculate GANs loss
        g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))
        #d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                                #labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

        #calculate the average of discriminator output for SR data
        E_of_SR = tf.reduce_mean(d_SR, axis=0)
        #calculate the average of discriminator output for HR data
        E_of_HR = tf.reduce_mean(d_HR, axis=0)
        # D_Ra_loss(xr, xf ) = σ(C(xr) − Exf[C(xf)]); D_Ra_loss(xf, xr ) = σ(C(xf) − Exr[C(xr)])
        d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR-E_of_SR, d_SR-E_of_HR], axis=0),
                                                                labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

        #total loss
        generator_loss = tf.math.reduce_mean(input_tensor=content_loss) + self.alpha_advers*tf.math.reduce_mean(input_tensor=g_advers_loss)
        discriminator_loss = tf.math.reduce_mean(input_tensor=d_advers_loss)
        
        return generator_loss, discriminator_loss
               
    def train_step(self, HR_batch, LR_batch):
        '''
        Define training step for GAN
        '''
        #record the computation of loss
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_SR = self.generator(LR_batch, training=True)
    
            d_HR = self.discriminator(HR_batch, training=True)
            d_SR = self.discriminator(generated_SR, training=True)

            gen_loss, disc_loss = self.loss_func(generated_SR, HR_batch, d_HR, d_SR)
        
        #compute gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        #initialize optimizer
        generator_optimizer, discriminator_optimizer = self.optimizer()
        
        #update gradients with optimizer
        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def trainCNN(self):
        #train CNN, only the generator
        print('Start Training CNN now.')
        file = open('CNN_epoch_loss.txt', 'a')
        for epoch in range(self.EPOCHS):
            start = time.time()

            HR_batched, LR_batched = self.get_batched()
            epoch_loss = 0. #record epoch loss
            for HR_batch, LR_batch in zip(HR_batched, LR_batched):
                with tf.GradientTape() as tape:
                    generated_SR = self.generator(LR_batch, training=True)
                    content_loss = tf.math.reduce_mean(input_tensor=tf.math.subtract(HR_batch, generated_SR)**2, axis=[1, 2, 3])
                    batch_loss = tf.reduce_mean(input_tensor=content_loss)

                #compute gradients
                gradients_of_generator = tape.gradient(batch_loss, self.generator.trainable_variables)
                #initialize optimizer
                generator_optimizer, _ = self.optimizer()
                #update gradients with optimizer
                generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                epoch_loss += batch_loss / self.BATCH_SIZE
            print("Generator loss at epoch {}: ".format(epoch+1) + str(epoch_loss.numpy()))
            file.write(str(epoch+1) + ' ' + str(epoch_loss.numpy()) + '\n')

            # Save the model every 1 epochs since its slow to train
            if (epoch + 1) % 5 == 0: #save model every 5 epoch
                # show images every epoch
                self.test_visualize_save_data(self.generator, epoch+1, self.seeds)
                checkpoint_prefix = os.path.join('./training_checkpoints', "ckpt")
                print('Model saved at epoch {}.'.format(epoch + 1))
                self.save_checkpoint().save(file_prefix = checkpoint_prefix)
    
            #print training time for each epoch
            print ('Time for training CNN at epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            
        file.close()
        
    def train(self, epochs_for_GAN=None):
        '''
        Varibale: epochs_for_GAN -- we can train additional different numbers of epochs for GANs
        '''
        if epochs_for_GAN == None:
            epochs_for_GAN = self.EPOCHS

        print('Start Training GANs now.')
        file = open('GAN_epoch_loss.txt', 'a')
        for epoch in range(epochs_for_GAN):
            start = time.time()

            HR_batched, LR_batched = self.get_batched()
            gen_epoch_loss, disc_epoch_loss = 0., 0. #record epoch loss
            for HR_batch, LR_batch in zip(HR_batched, LR_batched):
                gen_loss, disc_loss = self.train_step(HR_batch, LR_batch)
                gen_epoch_loss += gen_loss / self.BATCH_SIZE
                disc_epoch_loss += disc_loss / self.BATCH_SIZE
            print("Generator loss at epoch {}: ".format(epoch+1) + str(gen_epoch_loss.numpy()))
            print("Discriminator loss at epoch {}: ".format(epoch+1) + str(disc_epoch_loss.numpy()))
            file.write(str(epoch+1) + ' ' + str(gen_epoch_loss.numpy()) + ' ' + str(disc_epoch_loss.numpy()) + '\n')

            # Save the model every 1 epochs since its slow to train
            if (epoch + 1) % 5 == 0: #save model every 5 epoch
                self.test_visualize_save_data(self.generator, epoch+1, self.seeds)
                checkpoint_prefix = os.path.join('./training_checkpoints', "ckpt")
                print('Model saved at epoch {}.'.format(epoch + 1))
                self.save_checkpoint().save(file_prefix = checkpoint_prefix)
    
            #print training time for each epoch
            print ('Time for training GANs at epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        file.close()
        #make sure the final epoch is saved if epochs_for_GAN is different from self.EPOCHS
        display.clear_output(wait=True)
        self.test_visualize_save_data(self.generator, epochs_for_GAN, self.seeds, epochs_for_GAN)