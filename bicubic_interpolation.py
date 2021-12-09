# -*- coding: utf-8 -*-
"""
@author: GeeksforGeeks.org, edited by Shenbin Qian

"""

# import modules

import numpy as np
import math
import sys
import time
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import generate_LR


# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0


# Padding
def padding(img, H, W, C):
    zimg = np.zeros((H+4, W+4, C))
    zimg[2:H+2, 2:W+2, :C] = img
    
    # Pad the first/last two col and row
    zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
    zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
    zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]
    
    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
    zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
    zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
    return zimg


# Bicubic operation
def bicubic(img, ratio, a):

    # Get image size
    H, W, C = img.shape

    # Here H = Height, W = weight,
    # C = Number of channels if the
    # image is coloured.
    img = padding(img, H, W, C)

    # Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)

    # Converting into matrix
    dst = np.zeros((dH, dW, C))

    # np.zeroes generates a matrix
    # consisting only of zeroes
    # Here we initialize our answer
    # (dst) as zero

    h = 1/ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    start_time = time.time()
    for c in range(C):
        for j in range(dH):
            for i in range(dW):

                # Getting the coordinates of the
                # nearby values
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                # Considering all nearby 16 values
                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c],
                                    img[int(y-y2), int(x-x1), c],
                                    img[int(y+y3), int(x-x1), c],
                                    img[int(y+y4), int(x-x1), c]],
                                   [img[int(y-y1), int(x-x2), c],
                                    img[int(y-y2), int(x-x2), c],
                                    img[int(y+y3), int(x-x2), c],
                                    img[int(y+y4), int(x-x2), c]],
                                   [img[int(y-y1), int(x+x3), c],
                                    img[int(y-y2), int(x+x3), c],
                                    img[int(y+y3), int(x+x3), c],
                                    img[int(y+y4), int(x+x3), c]],
                                   [img[int(y-y1), int(x+x4), c],
                                    img[int(y-y2), int(x+x4), c],
                                    img[int(y+y3), int(x+x4), c],
                                    img[int(y+y4), int(x+x4), c]]])
                mat_r = np.matrix(
                    [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                
                # Here the dot function is used to get the dot
                # product of 2 matrices
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
                
        print('Channel {} completed!'.format(c+1))

    print('Converting takes %.2f seconds.' % (time.time() - start_time), flush=True)
    # If there is an error message, it
    # directly goes to stderr
    sys.stderr.write('\n')
    
    # Flushing the buffer
    sys.stderr.flush()
    return dst

def plot_output_data(data):
    vmin, vmax = np.min(data[:,:,:]), np.max(data[:,:,:])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis', origin='lower')
    plt.title('Bicubic interpolation SR data', fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax) 
    plt.show()

if __name__ == '__main__':
    # Read in data
    data_path='E:/ADSS/FinalProject504J/Data/msc_20200115_metdata.nc'
    raw_dataset = Dataset(data_path)
    temperature = raw_dataset["specific_humidity_0"][:560,:416,0,0].reshape(560, 416, 1, 1).transpose(3, 0, 1, 2)
    
    LR = generate_LR(temperature, factor=16).numpy().reshape(35, 26, 1)
    '''
    #if LR is more than one example
    #retrieve H W and C
    H, W, C = LR.shape
    #out_shape = [1, H*ratio, W*ratio, C]
    #out = np.zeros(out_shape, dtype=np.float64)
    # Loop over each LR example and passing them in the bicubic function
    for i in range(LR.shape[0]):
        if i == 0:
            out = np.add(out, bicubic(LR[i,:,:,:].reshape(H,W,C), ratio, a).reshape(out_shape))
        else:
            out = np.concatenate((out, bicubic(LR[i,:,:,:].reshape(H,W,C), ratio, a).reshape(out_shape)), axis=0)
    '''
    # Scale factor
    ratio = 4
    # Coefficient mostly between -0.5 to -0.75 for optimum performance
    a = -1/2
    out = bicubic(LR, ratio, a)
    print('Completed!')
    
    #plot data
    plot_output_data(out)
    #save data
    #np.save('bicubic_interp_SR.npy', out)
    #print('Interpolated SR data saved!')
    # display shapes of both HR and SR
    print('Original Image Shape:', LR.shape)
    print('Generated Bicubic Image Shape:', out.shape)
