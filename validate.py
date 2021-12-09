# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:10:32 2021

@author: Shenbin Qian
"""
from netCDF4 import Dataset
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import generate_LR, turn2origin_shape, get_Tpqh, refractivity, denormalise

class Validator():
    def __init__(self, SR_file_dir='E:/ADSS/FinalProject504J/SR_sample'):
        self.SR_file_dir = SR_file_dir

    def read_raw_data_slice(self, file='E:/ADSS/FinalProject504J/Data/msc_20200115_metdata.nc', height_index=70):
        #read raw HR data and return HR data in training shape
        atmos_data = Dataset(file)
        if height_index <= 70:
            temperature = tf.convert_to_tensor(atmos_data["air_temperature_0"][:560,:416,:height_index,0].reshape(560, 416, 1, height_index).transpose(3, 0, 1, 2))
            pressure = tf.convert_to_tensor(atmos_data["air_pressure"][:560,:416,:height_index,0].reshape(560, 416, 1, height_index).transpose(3, 0, 1, 2)) / 100
            humidity = tf.convert_to_tensor(atmos_data["specific_humidity_0"][:560,:416,:height_index,0].reshape(560, 416, 1, height_index).transpose(3, 0, 1, 2))
            return temperature, pressure, humidity
        else:
            print('Height index cannot be larger than 70!')

    def read_SR_data(self):
        #read all generated SR data given file directory
        for foldername, _, files in os.walk(self.SR_file_dir):
            for file in files:
                path = os.path.join(foldername, file)
                if 'humidity' in path:
                    humidity = tf.convert_to_tensor(np.load(path))
                elif 'pressure' in path:
                    pressure = tf.convert_to_tensor(np.load(path))
                elif 'temperature' in path:
                    temperature = tf.convert_to_tensor(np.load(path))
                else:
                    print('Wrong file directory!')
        return temperature, pressure, humidity

    def choose_points(self, data, example_index=0, point1=(90, 65), point2=(18,130), point3=(50, 40)):
        '''
        Visualise the points chosen on the grid
        Variable: data needs to be in training shape of (number_of_example, lon, lat, 1)
        '''
        n, h, w, c = data.shape
        vmin, vmax = np.min(data[example_index,:,:,:]), np.max(data[example_index,:,:,:])
    
        circle1 = plt.Circle(point1, 0.9, color='r')
        circle2 = plt.Circle(point2, 0.9, color='r')
        circle3 = plt.Circle(point3, 0.9, color='r')
        plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        im = ax.imshow(tf.reshape(data[example_index,:, :, :], (h,w,c)), vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
        plt.title('Data at example{}'.format(example_index), fontsize=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
        plt.show()
    
    def viz_vertical_changes(self, data, point=(65, 90)):
        '''
        Pass a point to visualse vertical changes
        Variable data needs to be in original shape of (lon, lat, height, timestep=1)
        '''
        h, w = point
        v = data[h, w, :, :].numpy().reshape(-1)
        
        plt.figure(figsize=[10,8])
        plt.plot(v, 'r-')
        plt.xlabel('Height Level')
        plt.title('Vertical changes at point ({}, {})'.format(str(h), str(w)))
        plt.show()
    
    def compare_vertical_changes(self, HR, SR, point=(65, 90)):
        '''
        Pass a point and HR, SR data to compare vertical changes after super-resolution
        Variable data needs to be in original shape of (lon, lat, height, timestep=1)
        '''
        h, w = point
        HR_line = HR[h, w, :, :].numpy().reshape(-1)
        SR_line = SR[h, w, :, :].numpy().reshape(-1)
        
        fig = plt.figure(figsize=[10,8])
        #add subplot
        ax = fig.add_subplot(111)
        ax.plot(HR_line, 'b-', label='HR_data')
        #add twin x
        ax2 = ax.twinx()
        ax2.plot(SR_line, 'r-', label='SR_data')
        ax.set_xlabel('Height Level')
        ax.set_ylabel('HR vertical changes')
        ax2.set_ylabel('SR vertical changes')
        plt.title('Vertical changes of HR and SR at chosen points')
        fig.legend()
        plt.show()
        
    def plot_prediction_intervals(self, temp_denorm, MR_temp, point=(65, 90)):
        '''
        Pass in denormalised temperature data and MR data to
        plot Gaussian prediction intervals, point is optional
        '''
        h, w = point
        std_list = []
        for i in range(temp_denorm.shape[0]):
            std_list.append(np.std(temp_denorm[i, :, :, :].numpy().reshape(-1)))
        # 95% confidence interval
        interval = 1.96 * np.array(std_list)
        lower = np.squeeze(temp_denorm[:, h, w, :].numpy()) - interval 
        upper = np.squeeze(temp_denorm[:, h, w, :].numpy()) + interval 

        real = MR_temp[:, h, w, :].numpy().reshape(-1)  
        sr = temp_denorm[:, h, w, :].numpy().reshape(-1)
        #plot HR, SR and prediction intervals
        plt.figure(figsize=[8,6])
        plt.plot(real, 'b-', label='HR data')
        plt.plot(sr, 'r-', label='SR data')
        plt.plot(upper, 'g-', label='Upper')
        plt.plot(lower, 'g-', label='Lower')
        plt.legend()
        plt.xlabel('Height index')
        plt.ylabel('Temperature in Kelvin(K)')
        plt.title('Vertical changes of temperature at Point ({}, {})'.format(str(h), str(w)))
        plt.show()
    
    def compare_SR2HR_on_point(self, point=None, height_index=0, vali_target='temperature'):
        '''
        Parameters
        ----------
        point : TYPE, optional
            The point on the grid for validation. The default is None, for which we calculate the mean and std.
        height_index : TYPE, optional
            The height level. The default is 0.
        vali_target : TYPE, optional
            Which data to validate, temperature, pressure or humidity. The default is 'temperature'.

        Returns
        -------
        print HR and SR refractivity results
        '''
        # deal with original 'HR' data
        raw_temp, raw_pres, raw_hum = self.read_raw_data_slice()
        MR_temp = turn2origin_shape(generate_LR(raw_temp,factor=4))
        MR_pres = turn2origin_shape(generate_LR(raw_pres,factor=4))
        MR_hum = turn2origin_shape(generate_LR(raw_hum,factor=4))
        
        MR_T, MR_p, MR_q, MR_h = get_Tpqh(MR_temp, MR_pres, MR_hum, height_index=height_index)
        MR_grid = refractivity(MR_T, MR_p, MR_q, MR_h)
        
        # mu and variance of the whole dataset
        temp_mu_var =[245.169, 700.7318]
        pres_mu_var = [399.27563, 127388.86]
        hum_mu_var = [0.0015283022, 8.156426e-06]
        
        # deal with SR data
        SR_temp_train, SR_pres_train, SR_hum_train = self.read_SR_data()
        SR_temp = denormalise(turn2origin_shape(SR_temp_train),temp_mu_var)
        SR_pres = denormalise(turn2origin_shape(SR_pres_train), pres_mu_var)
        SR_hum = denormalise(turn2origin_shape(SR_hum_train), hum_mu_var)

        if point:
            h,w = point #retrieve x and y index
            if vali_target == 'all':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(SR_temp, SR_pres, SR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print(MR_grid[h, w, height_index, 0])
                print(SR_grid[h, w, height_index, 0])
            elif vali_target == 'temperature':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(SR_temp, MR_pres, MR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print(MR_grid[h, w, height_index, 0])
                print(SR_grid[h, w, height_index, 0])
            elif vali_target == 'pressure':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(MR_temp, SR_pres, MR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print(MR_grid[h, w, height_index, 0])
                print(SR_grid[h, w, height_index, 0])
            elif vali_target == 'humidity':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(MR_temp, MR_pres, SR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print(MR_grid[h, w, height_index, 0])
                print(SR_grid[h, w, height_index, 0])
            else:
                print('Validation target needs to be "temperature", "pressure", "humidity" or "all"!')
        #if no point passed, calculate the mean and standard deviation of the whole grid
        else:
            print('The mean of the original data is '+ str(tf.math.reduce_mean(MR_grid).numpy()))
            print('The standard deviation of the original data is '+ str(tf.math.reduce_std(MR_grid).numpy()))
            if vali_target == 'all':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(SR_temp, SR_pres, SR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print('The mean of the SR data is '+ str(tf.math.reduce_mean(SR_grid).numpy()))
                print('The standard deviation of the SR data is '+ str(tf.math.reduce_std(SR_grid).numpy()))
            elif vali_target == 'temperature':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(SR_temp, MR_pres, MR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print('The mean of the SR data is '+ str(tf.math.reduce_mean(SR_grid).numpy()))
                print('The standard deviation of the SR data is '+ str(tf.math.reduce_std(SR_grid).numpy()))
            elif vali_target == 'pressure':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(MR_temp, SR_pres, MR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print('The mean of the SR data is '+ str(tf.math.reduce_mean(SR_grid).numpy()))
                print('The standard deviation of the SR data is '+ str(tf.math.reduce_std(SR_grid).numpy()))
            elif vali_target == 'humidity':
                SR_T, SR_p, SR_q, SR_h = get_Tpqh(MR_temp, MR_pres, SR_hum, height_index=height_index)
                SR_grid = refractivity(SR_T, SR_p, SR_q, SR_h)
                print('The mean of the SR data is '+ str(tf.math.reduce_mean(SR_grid).numpy()))
                print('The standard deviation of the SR data is '+ str(tf.math.reduce_std(SR_grid).numpy()))
            else:
                print('Validation target needs to be "temperature", "pressure", "humidity" or "all"!')

class AverageM():
    '''
    Define a class to plot average refractivity of all points for all types of data against height level
    '''
    def __init__(self, file="E:/ADSS/FinalProject504J/Data/m_for_height.csv"):
        self.file = file # where average m is calculated by functions in Validator
        self.plot_m_temperature = self.plot_m_temperature()
        self.plot_m_pressure = self.plot_m_pressure()
        self.plot_m_humidity = self.plot_m_humidity()
        
    def read_m(self):
        m = pd.read_csv(self.file)
        return m
    
    def plot_m_temperature(self):
        m = self.read_m()
        plt.figure(figsize=[8,6])
        plt.plot(m['Height_level_index'], m['Average_m_for_original_data'], color='blue', label='m for original data')
        plt.plot(m['Height_level_index'], m['Average_m_for_SR_temperature'], color='red', label='m for SR temperature')
        plt.legend()

        plt.title('Average Refractivity for Original and SR Temperature')
        plt.xlabel('Height Index')
        plt.ylabel('Refractivity')
        plt.show()
        
    def plot_m_pressure(self):
        m = self.read_m()
        plt.figure(figsize=[8,6])
        plt.plot(m['Height_level_index'], m['Average_m_for_original_data'], color='blue', label='m for original data')
        plt.plot(m['Height_level_index'], m['Average_m_for_SR_pressure'], color='red', label='m for SR pressure')
        plt.legend()

        plt.title('Average Refractivity for Original and SR Pressure')
        plt.xlabel('Height Index')
        plt.ylabel('Refractivity')
        plt.show()
        
    def plot_m_humidity(self):
        m = self.read_m()
        plt.figure(figsize=[8,6])
        plt.plot(m['Height_level_index'], m['Average_m_for_original_data'], color='blue', label='m for original data')
        plt.plot(m['Height_level_index'], m['Average_m_for_SR_humidity'], color='red', label='m for SR humidity')
        plt.legend()

        plt.title('Average Refractivity for Original and SR Humidity')
        plt.xlabel('Height Index')
        plt.ylabel('Refractivity')
        plt.show()

if __name__ == '__main__':
    #instantiate a validator
    validator = Validator(SR_file_dir='./SR_sample')
    #compare SR to HR
    validator.compare_SR2HR_on_point()
    
    #extract a slice of raw data
    HR_temp, HR_pres, HR_hum = validator.read_raw_data_slice(file='msc_20200115_metdata.nc')
    MR_temp = generate_LR(HR_temp,factor=4) #generate MR
    SR_temp, SR_pres, SR_hum = validator.read_SR_data() #read SR data
    
    # mu and variance
    temp_mu_var =[245.169, 700.7318]
    pres_mu_var = [399.27563, 127388.86]
    hum_mu_var = [0.0015283022, 8.156426e-06]
        
    validator.compare_vertical_changes(turn2origin_shape(MR_temp), turn2origin_shape(denormalise(SR_temp, temp_mu_var)))
    
    #plot SR prediction intervals of temperature data
    validator.plot_prediction_intervals(denormalise(SR_temp, temp_mu_var), MR_temp)

    #plot distribution of SR pressure
    re = SR_pres.numpy().reshape(-1)
    plt.hist(re)
    plt.title('Distribution of generated SR pressure')
    plt.show()
    
    # uncomment to plot average refractivity of all points for all types of data against height level
    #AverageM()
