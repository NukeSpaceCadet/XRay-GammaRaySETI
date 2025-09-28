# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:29:57 2024, revised April 2025, July 2025

Code for analyzing normalized information content of alien 
x-ray/gamma-ray laser signals

April 2025, revised to call OpenMC model of NaI detector model with high fidelity

NOTE: natural signals detected in space would not have sharp peaks, so this is another 
filter for artificial signals

@author: Lucas Beveridge
"""

import numpy as np # for CPU array operations
# import cupy as cp # for GPU offloading of array operations
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.stats import entropy # for CPU entropy operations
# from cupyx.scipy.stats import entropy # for GPU offloading of entropy operations
# import time
import os
import math
from OpenMC_detector_with_background import spectrum_maker
import subprocess
import pickle
import random

# constants
pulse_duration = 1e-13 # seconds
detector_interval = 1e-8 # cycle time for detector (rise-time and dead-time), seconds
num_channels = 2**12 # 4096 channels
signal_photon_energy = 414 # gamma-ray energy in keV

###############################################################################
# This function generates a sequence of spectra based on the time-dependent pulse-train

def signal_maker(input_pulse_array, interval, pulse_energy, name):
    times = input_pulse_array[0] # extract time indices from incoming pulse train
    pulse_mag = input_pulse_array[1] # extract magnitude of gamma ray in pulse train (can be 0)
    
    dt = times[1]-times[0] # extract duration of individual pulses

    count = 0        
    timer = 0
    pulse_count = 0
    pulses_per_DT = np.ceil(interval/dt)-1 # pulses per detector interval
    # print(pulses_per_DT)
    limit = times[len(times)-1] # maximum signal duration, extracted from input
    dead_mult = np.ceil(limit/interval)-1 # the number of detector intervals in the signal duration
    # print(dead_mult)
    
    if pulses_per_DT < 1:
        print("Warning! Duration between pulses is longer than detector interval")
        
    time = [] # initialize time stamp array for spectra
    spectra_array = [] # intialize spectra array
    
    for j in range(dead_mult.astype(int)): # step over detector intervals, round to nearest integer

        pulse_count = sum(pulse_mag[((j)*abs(pulses_per_DT.astype(int))):((j+1)*abs(pulses_per_DT.astype(int)))])
        # print(spectra_maker(pulse_energy,pulse_count,max_spec_ener,SN_ratio,0,bin_width)[1])

        # Create a spectrum with a magnitude equal to the total pulse count:
        # spectrum, errors, lower_bounds, energies = spectrum_maker(pulse_energy, 0.07, 2**12, j, pulse_count, name) # run OpenMC photon transport model
        
        with open("input.pkl", "wb") as f:
            pickle.dump((pulse_energy, pulse_count, name, num_channels, j), f)
        # Run the OpenMC worker
        subprocess.run(["python", "run_openmc_single.py"], check=True)
        
        # Load and use results
        with open("output.pkl", "rb") as f:
            spectrum, errors, lower_bounds, energies = pickle.load(f)
        
        timer = timer + interval # keep track of real-time
        time.append(timer) # place time stamp into array
        spectra_array.append(spectrum)
    
    ratio = interval/dt # calculate the ratio of the detector interval to time-step between pulses
    
    return (spectra_array,time,ratio)

###############################################################################
def get_binary_file(filename, step_size):
    # open binary file and generate array of 
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        
    things = [bin(i)[2:] for i in fileContent] # convert raw file values to list of binary numbers
    binary_data = "".join(things) # flatten list of numbers into a single number (like serial data)
    binary_list = [int(i) for i in binary_data] # split single number into array of 1's and 0's
    length = len(binary_list)
    duration = step_size*length # duration of signal
    time_arr = np.linspace(0,duration,length, dtype = float) # create time step list for signal
    signal_arr = (time_arr,binary_list) # combine lists for time and binary numbers for output array
    return signal_arr, length, time_arr

###############################################################################
# Function to compute relative entropy for an array column

def compute_kl_divergence(pk_column, qk_column):
    # Normalize pk and qk to get probability distributions (if needed)
    pk_probs = pk_column / pk_column.sum()
    qk_probs = qk_column / qk_column.sum()
    
    # Compute KL divergence for this column
    return entropy(pk_column, qk_column)
###############################################################################

# Generate spectra for actual signal
signal, length, time_arr = get_binary_file("dlc2.jpg", pulse_duration)
spec_arr,time,ratio = signal_maker(signal, detector_interval, signal_photon_energy, "signal")
stacked_signal_specs = np.vstack(spec_arr)

# Generate spectra for constant signal (all 1's)
constant_signal = [1]*length
const_specs, time_const, ratio2 = signal_maker((time_arr, constant_signal), detector_interval, signal_photon_energy, "all_ones")
stacked_const_specs = np.vstack(const_specs)
number_of_intervals = np.floor(length*(pulse_duration/detector_interval)) # total number of detector intervals during duration of the signal

# Generate 100 random spectra and average them
averaged_spectra_for_this_time_step = []
for spec_index in range(int(number_of_intervals)):  
    rand_step=[]
    for number_of_random_specs in range(100):
        rand_spectrum, rand_errors, lower_bounds, energies = spectrum_maker(random.uniform(0, 1000), 0.07, num_channels, spec_index, np.random.randint(0, round(detector_interval/pulse_duration,0)+1), "Random")
        rand_step.append(rand_spectrum)
    averaged_spectra_for_this_time_step.append([sum(values) / len(values) for values in zip(*rand_step)])
stacked_rand_specs1= np.vstack(averaged_spectra_for_this_time_step)

# Generate 100 random spectra and average them for another reference
averaged_spectra_for_this_time_step2 = []
for spec_index in range(int(number_of_intervals)):  
    rand_step=[]
    for number_of_random_specs in range(100):
        rand_spectrum, rand_errors, lower_bounds, energies = spectrum_maker(random.uniform(0, 1000), 0.07, num_channels, spec_index, np.random.randint(0, round(detector_interval/pulse_duration,0)+1), "Random")
        rand_step.append(rand_spectrum)
    averaged_spectra_for_this_time_step2.append([sum(values) / len(values) for values in zip(*rand_step)])
stacked_rand_specs2= np.vstack(averaged_spectra_for_this_time_step2)

# Generate 100 random spectra and average them for yet another reference
averaged_spectra_for_this_time_step3 = []
for spec_index in range(int(number_of_intervals)):  
    rand_step=[]
    for number_of_random_specs in range(100):
        rand_spectrum, rand_errors, lower_bounds, energies = spectrum_maker(random.uniform(0, 1000), 0.07, num_channels, spec_index, np.random.randint(0, round(detector_interval/pulse_duration,0)+1), "Random")
        rand_step.append(rand_spectrum)
    averaged_spectra_for_this_time_step3.append([sum(values) / len(values) for values in zip(*rand_step)])
stacked_rand_specs3= np.vstack(averaged_spectra_for_this_time_step3)

M_signal = [] # normalized statistic of information content for real signal
for column in range(len(stacked_const_specs[0,:])):
    KL_signal = compute_kl_divergence(stacked_signal_specs[:,column], stacked_rand_specs1[:,column])
    KL_const = compute_kl_divergence(stacked_const_specs[:,column], stacked_rand_specs1[:,column])
    KL_rand = compute_kl_divergence(stacked_rand_specs2[:,column], stacked_rand_specs1[:,column])
    M_signal.append(abs((KL_signal - KL_rand) / (KL_rand- KL_const))) # Normalized information content (from 0 to 1)

M_random = [] # normalized statistic of information content for noise
for column in range(len(stacked_const_specs[0,:])):
    KL_signal = compute_kl_divergence(stacked_rand_specs3[:,column], stacked_rand_specs1[:,column])
    KL_const = compute_kl_divergence(stacked_const_specs[:,column], stacked_rand_specs1[:,column])
    KL_rand = compute_kl_divergence(stacked_rand_specs2[:,column], stacked_rand_specs1[:,column])
    M_random.append(abs((KL_signal - KL_rand) / (KL_rand - KL_const))) # Normalized information content (from 0 to 1)

# Plot both lists
plt.plot(M_signal, label='Artificial Signal', color='blue')
plt.plot(M_random, label='Noise', color='red')

# Add legend and labels
plt.xlabel('Detector bin energy (keV)')
plt.ylabel('Normalized information content')
# plt.title('Overlayed Lists')
plt.legend()

# Show plot
plt.show()