# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:53:32 2021

@author: Shenbin Qian
"""

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc, os


def read_raw_data_for_train(file_dir="C:/Users/sq235", extract_data='temperature'):
    '''
    Read data from .nc file for training by passing file directory and extract-data name
    Parameters
    ----------
    file_dir : string, optional
        The default is "C:/Users/sq235".
    extract_data : string, optional
        The default is 'temperature'.

    Returns
    -------
    Tensor of HR data with shape (n, lon, lat, 1), n = height * timestep

    '''
    #extract temperature
    data_list = []
    if extract_data == 'temperature':
        for foldername, _, filenames in os.walk(file_dir):
            for file in filenames:
                if file.endswith('.nc'):
                    absPath = os.path.abspath(os.path.join(foldername, file))
                    print('Reading in {}...'.format(file))
                    atmos_data = Dataset(absPath)
                    temperature = np.array(atmos_data["air_temperature_0"][:560,:416,:,:]).reshape(560, 416, 1, 70*12).transpose(3, 0, 1, 2)
                    #delete unused variable for saving RAM
                    del atmos_data
                    gc.collect()
                    data_list.append(temperature)
        print("All {} data read in!".format(extract_data))
        return tf.convert_to_tensor(np.concatenate(data_list, axis=0))
    #extract pressure
    elif extract_data == 'pressure':
        for foldername, _, filenames in os.walk(file_dir):
            for file in filenames:
                if file.endswith('.nc'):
                    absPath = os.path.abspath(os.path.join(foldername, file))
                    print('Reading in {}...'.format(file))
                    atmos_data = Dataset(absPath)
                    pressure = np.array(atmos_data["air_pressure"][:560,:416,:,:]).reshape(560, 416, 1, 70*12).transpose(3, 0, 1, 2)
                    del atmos_data
                    gc.collect()
                    data_list.append(pressure)
        print("All {} data read in!".format(extract_data))
        return tf.convert_to_tensor(np.concatenate(data_list, axis=0)) / 100
    #extract humidity
    elif extract_data == 'humidity':
        for foldername, _, filenames in os.walk(file_dir):
            for file in filenames:
                if file.endswith('.nc'):
                    absPath = os.path.abspath(os.path.join(foldername, file))
                    print('Reading in {}...'.format(file))
                    atmos_data = Dataset(absPath)
                    humidity = np.array(atmos_data["specific_humidity_0"][:560,:416,:,:]).reshape(560, 416, 1, 70*12).transpose(3, 0, 1, 2)
                    del atmos_data
                    gc.collect()
                    data_list.append(humidity)
        print("All {} data read in!".format(extract_data))
        return tf.convert_to_tensor(np.concatenate(data_list, axis=0))
    #extract all data
    elif extract_data == 'All':
        for foldername, _, filenames in os.walk(file_dir):
            for file in filenames:
                if file.endswith('.nc'):
                    absPath = os.path.abspath(os.path.join(foldername, file))
                    print('Reading in {}...'.format(file))
                    atmos_data = Dataset(absPath)
                    temperature = np.array(atmos_data["air_temperature_0"][:560,:416,:,:]).reshape(560, 416, 1, 70*12).transpose(3, 0, 1, 2)
                    pressure = np.array(atmos_data["air_pressure"][:560,:416,:,:]).reshape(560, 416, 1, 70*12).transpose(3, 0, 1, 2)
                    humidity = np.array(atmos_data["specific_humidity_0"][:560,:416,:,:]).reshape(560, 416, 1, 70*12).transpose(3, 0, 1, 2)
                    data_list.append(temperature)
                    data_list.append(pressure)
                    data_list.append(humidity)
                    del atmos_data, temperature, pressure, humidity
                    gc.collect()
        print("All data read in!")
        return tf.convert_to_tensor(np.concatenate(data_list, axis=0))

    else:
        print('Extract_data should be "temperature", "pressure", "humidity" or "All"!')

def normalise(data):
    '''
    Parameters
    ----------
    data : Tensor to be normalised

    Returns
    -------
    normalised tensor
    '''
    mu, variance = tf.nn.moments(data, axes=[0, 1, 2, 3])
    print('Input data normalising...')
    return (data - mu) / tf.math.sqrt(variance)

def denormalise(data, mu_var):
    '''
    Parameters
    ----------
    data : Normalised tensor
    mu_var : list of two, mean and variance
    
    Returns
    -------
    denormalised data
    '''
    return data * tf.math.sqrt(mu_var[1]) + mu_var[0]

def generate_LR(original_data, factor=16):
    #Use average pooling layer to shrink by 1/factor
    return AveragePooling2D(pool_size=(factor, factor))(original_data)

def average_keep_dim(data_train, r=16):
    '''
    Similar function as generate_LR, but different effect: keep dimension of original dataset,
    but use repeated same value to replace orginal value. 
    This function mainly used for processing data before plotting map

    Parameters
    ----------
    data_train : tensor in training shape of (840, 560, 416, 1)
    r : TYPE, optional. The default is 16.
        DESCRIPTION. How much times to shrink

    Returns
    -------
    out : tensor of original HR dimension, but averaged value
    '''
    flat = tf.keras.layers.Flatten()(data_train).numpy()
    num = int(flat.shape[1] / r)
    for i in range(flat.shape[0]):
        n = 0
        for _ in range(num):
            avg = np.mean(flat[i,n:n+r])
            flat[i,n:n+r] = avg
            n += r
    out = tf.convert_to_tensor(flat.reshape(840, 560, 416, 1))
    return out

def turn2train_shape(original_data):
    #turn original data set into a shape used in tensorflow [n, h, w, c]
    lon, lat, height, ts = original_data.shape
    return tf.transpose(tf.reshape(original_data, (lon, lat, 1, height * ts)),(3,0,1,2))

def turn2origin_shape(training_data):
    #turn training data set back into the shape of original data for caculating refrectivity
    n, lon, lat, _ = training_data.shape
    #n is equal or larger than 840, the data set is equal or larger than one .nc file
    if n >= 840:
        if n % 840 == 0:
            f = int(n / 840)
            if f == 1: #data set of the size of 1 .nc file
                return tf.transpose(tf.reshape(training_data, (70, 12, lon, lat)),(2,3,0,1))
            if f == 2: #data set of the size of 2 .nc file
                ele1 = tf.transpose(tf.reshape(training_data[:840,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele2 = tf.transpose(tf.reshape(training_data[840:,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                return tf.concat([ele1, ele2], axis=3)
            if f == 3: #data set of the size of 3 .nc file
                ele1 = tf.transpose(tf.reshape(training_data[:840,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele2 = tf.transpose(tf.reshape(training_data[840:840*(f-1),:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele3 = tf.transpose(tf.reshape(training_data[840*(f-1):,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                return tf.concat([ele1, ele2, ele3], axis=3)
            if f == 4: #data set of the size of 4 .nc file
                ele1 = tf.transpose(tf.reshape(training_data[:840,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele2 = tf.transpose(tf.reshape(training_data[840:840*(f-2),:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele3 = tf.transpose(tf.reshape(training_data[840*(f-2):840*(f-1),:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele4 = tf.transpose(tf.reshape(training_data[840*(f-1):,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                return tf.concat([ele1, ele2, ele3, ele4], axis=3)
            if f == 5: #data set of the size of 5 .nc file
                ele1 = tf.transpose(tf.reshape(training_data[:840,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele2 = tf.transpose(tf.reshape(training_data[840:840*(f-3),:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele3 = tf.transpose(tf.reshape(training_data[840*(f-3):840*(f-2),:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele4 = tf.transpose(tf.reshape(training_data[840*(f-2):840*(f-1),:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                ele5 = tf.transpose(tf.reshape(training_data[840*(f-1):,:,:,:], (70, 12, lon, lat)),(2,3,0,1))
                return tf.concat([ele1, ele2, ele3, ele4, ele5], axis=3)
        else:
            print("Check dimensionality!")
    elif n <= 70:
        #if n is equal or smaller than 70, timestep is 1 and height is n
        return tf.transpose(tf.reshape(training_data, (n, 1, lon, lat)),(2,3,0,1))
    else:
        f = divmod(n, 70) #get quotient and remainder
        #quotient f[0] plus 1 is the dimensionality of timestep and remainder is height
        return tf.transpose(tf.reshape(training_data, (f[1], f[0]+1, lon, lat)),(2,3,0,1))

def plot_map(data, height_index=0, timestep=0):
    '''
    Parameters
    ----------
    data : original shape of (560, 416, height, timestep)
    height_index : TYPE, optional. The default is 0.
    timestep : TYPE, optional. The default is 0.
    '''
    data = data[:,:,height_index, timestep]
    atmos = Dataset('E:/ADSS/FinalProject504J/Data/msc_20200115_metdata.nc')
    longitudes = atmos["longitude"][:560]
    latitudes = atmos["latitude"][:416]
    plt.figure(figsize=(10, 8))
    m = Basemap(projection='cyl',resolution='l',
                llcrnrlat=min(latitudes), urcrnrlat=max(latitudes),
                llcrnrlon=min(longitudes), urcrnrlon=max(longitudes))
    plt.rcParams['font.size'] = '17'
    m.drawcoastlines()
    cf0 = m.pcolormesh(longitudes, latitudes, data.numpy().transpose(1,0), latlon=True)
    cbar0 = plt.colorbar(cf0,fraction=0.023, pad=0.04, label='hpa')
    cbar0.ax.tick_params(labelsize=10)
    plt.title("Air temperature map")
    plt.show()

def visualize_data_origin_shape(data, height_index=0, ts=0):
    '''
    Pass in two arguments, height index and timestep to visualise the data
    '''
    h, w, _, _ = data.shape
    vmin, vmax = np.min(data[:,:,height_index,ts]), np.max(data[:,:,height_index,ts])

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    im = ax.imshow(tf.reshape(data[:, :, height_index,ts], (h,w,1)), vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
    plt.title('Data at height index {} time step {}'.format(height_index, ts), fontsize=8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

def visualize_data_train_shape(data, example_index=0):
    '''
    Pass in an argument, number of example to visualise the data
    '''
    n, h, w, c = data.shape
    vmin, vmax = np.min(data[example_index,:,:,:]), np.max(data[example_index,:,:,:])

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    im = ax.imshow(tf.reshape(data[example_index,:, :, :], (h,w,c)), vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
    plt.title('Data at example{}'.format(example_index), fontsize=8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

def parse_CNN_epoch_loss(file):
    #parse epoch and loss in the file recorded during training for plotting
    epoch_num, loss = [], []
    with open(file, 'r') as f:
        results = f.readlines()
    for i in results:
        epoch_num.append(int(i.split()[0]))
        loss.append(float(i.split()[1]))
    return epoch_num, loss
    
def plot_CNN_epoch_loss(file):
    #plot CNN MSE loss against epoch
    epoch, loss = parse_CNN_epoch_loss(file)
    plt.plot(epoch, loss)
    plt.title('MSE loss for every epoch')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.show()

def parse_GAN_epoch_loss(file):
    #parse epoch and loss in the file recorded during training for plotting
    epoch_num, gen_loss, disc_loss = [], [], []
    with open(file, 'r') as f:
        results = f.readlines()
    for i in results:
        epoch_num.append(int(i.split()[0]))
        gen_loss.append(float(i.split()[1]))
        disc_loss.append(float(i.split()[2]))
    return epoch_num, gen_loss, disc_loss

def plot_GAN_epoch_loss(file):
    #plot GAN loss against epoch
    epoch, gen_loss, disc_loss = parse_GAN_epoch_loss(file)
    fig = plt.figure(figsize=[10,8])
    #add subplot
    ax = fig.add_subplot(111)
    ax.plot(epoch, gen_loss, 'b-', label='gen_loss')
    #add twin x
    ax2 = ax.twinx()
    ax2.plot(epoch, disc_loss, 'r-', label='disc_loss')
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Generator loss')
    ax2.set_ylabel('Discriminator loss')
    plt.title('GAN loss for every epoch')
    fig.legend()
    plt.show()

def refractivity(T, p, q, h):
    '''
    Parameters
    ----------
    T : Tensor of shape (lon, lat, 1, 1)
        temperature at certain height and timestep
    p : Tensor of shape (lon, lat, 1, 1)
        pressure at certain height and timestep
    q : Tensor of shape (lon, lat, 1, 1)
        humidity at certain height and timestep
    h : scalar
        height

    Returns
    -------
    m : Tensor of shape (lon, lat, 1, 1)
        refrectivity at certain height and timestep

    '''
    if T.shape == (560, 416, 1, 1) or T.shape == (140, 104, 1, 1) or T.shape == (35, 26, 1, 1):
        R = 6371000
        r_epsilon = 0.62198
        e = q * p / (r_epsilon + (1-r_epsilon)*q)
        m_dry = 77.6 * (p / T)
        m_moist = 373256 * e / T ** 2
        print("Calculating m ...")
        m = m_dry + m_moist + 10**6 * (h / R)
        return m
    else:
        print('You need to choose a certain height and timestep!')

def get_Tpqh(temperature, pressure, humidity, height_index=0, timestep_index=0):
    '''
    Parameters
    ----------
    temperature : Tensor of shape (lon, lat, height, timestep)
    pressure : Tensor of shape (lon, lat, height, timestep)
    humidity : Tensor of shape (lon, lat, height, timestep)
    height_index : scalar
    timestep_index : scalar

    Returns
    -------
    T : Tensor of shape (lon, lat, 1, 1)
        temperature at certain height and timestep
    p : Tensor of shape (lon, lat, 1, 1)
        pressure at certain height and timestep
    q : Tensor of shape (lon, lat, 1, 1)
        humidity at certain height and timestep
    h : scalar
        height
    '''
    if temperature.ndim == 4 and height_index < temperature.shape[2] and timestep_index < temperature.shape[3]:
        atmos = Dataset('E:/ADSS/FinalProject504J/Data/msc_20200115_metdata.nc')
        heights = atmos["level_height"][:]
        del atmos
        gc.collect()
        #heights = air_temperature.coord("level_height").points
        lon,lat,_,_ = temperature.shape
        print('Data prepared!')
        T = tf.reshape(temperature[:,:,height_index,timestep_index], (lon,lat,1,1))
        p = tf.reshape(pressure[:,:,height_index,timestep_index], (lon,lat,1,1))
        q = tf.reshape(humidity[:,:,height_index,timestep_index], (lon,lat,1,1))
        h = heights[height_index]
        return T, p, q, h
    else:
        print('Incorrect input arguments!')
        