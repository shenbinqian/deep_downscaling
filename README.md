# deep_downscaling

This is the MSc thesis project of Shenbin Qian with the UK Met Office titled *Deep-learning Based Downscaling of Numerical Weather Prediction Data*

The main target of this project is to use low-resolution (LR) numerical weather prediction (NWP) data to generate high-resolution (HR) data using deep learning techniques, especially single image super resolution (SR) techniques like SRCNN and SRGAN.

This project implements and improves the SRGAN model and uses the relativistic discriminator in the ESRGAN to super-resolve LR temperature, pressure and humidity data from the Met Office.

## Code

**The main.py** is the main programme that reads the data and runs the whole model. 

**The main_restore.py** can be used to restore model for transfer learning or continuing training.

**The networks.py** is where our model networks are defined. 

**The trainer.py** defines a class to set up hyperparameters and training details. 

**The utils.py** is where some basic functions like normalisation are defined. 

**The test.py** is used to restore the trained model and generate test results. 

**The validate.py** is used to validate SR results through the refractivity equation modified by the Met Office in "Met_office_data_validation.pdf". 

**The bicubic_interpolation.py** is used to generate bicubic interpolated results.

## Data, models and others

Since the original HR data from the Met Office cannot be published here, we simulated some samples in the directory **data** as a "toy" dataset for displaying the data structure as well as kicking off the model setup. 

**The temperature_simulate.npy** contains simulated HR temperature data with the dimensionality of (560, 416, 5, 1) representing longitude, latitude, height and timestep, which could be reshaped into (5, 560, 416, 1) during training.

**The pressure_simulate.npy** contains simulated HR pressure data with the dimensionality of (560, 416, 5, 1) representing longitude, latitude, height and timestep, which could be reshaped into (5, 560, 416, 1) during training.

 **The humidity_simulate.npy** contains simulated HR humidity data with the dimensionality of (560, 416, 5, 1) representing longitude, latitude, height and timestep, which could be reshaped into (5, 560, 416, 1) during training.

Data in **SR_sample** are samples of generated SR temperature, pressure and humidity.

In **trained_models**, we saved our models trained on temperature, pressure and humidity data as well as on the concatenated data of all three parameters. These models can be restored for validation or transfer learning.

**The m_for_height.csv** contains the average refractivity values of all the points on the SR grid against height index for validation purposes.