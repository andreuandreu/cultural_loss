import string
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import unittest
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy


'''
what would be, eventually, a trait decay inside a knode, this code shall be eficient, fast, repeated

called "drifwood model" as it tries to model what is the threshold that the presence of a resouce/dependence
has to occur in order for the trait to remain "knowlwgelable" onside of the knode

'''

def while_loop():
    threshold = 0.5
    k = [1]
    i = 0
    while i < 10 and k[i] > threshold:
        
        k.append( k[i-1] - 0.1)
        i = i + 1
        print(i, k)

def trait_evol( stocastic_dependence, decay, time_step, k0, threshold = 1/10):
    
    kt = [k0]
    time_vec = [time_step]
    i = 0
    while i < len(stocastic_dependence)-1:
        
        t = time_step
        #if i > 988:
            #print('iii', i, t)
        while i < len(stocastic_dependence)-1 and stocastic_dependence[i] == 0 and kt[i] > threshold:
            t = t + time_step
            i = i+1
            kt.append(kt[i-1]*np.exp(-t*decay))
            time_vec.append(t)
            if i > 998:
                print('tttt it ended!', t, kt[i])
            
        if kt[i] > threshold: 
            kt.append(k0)
            time_vec.append(time_step)
        else: 
            #print("end of loop at ", i, kt[i], time_vec[i])
            break

        i = i+1
    
    return kt , time_vec


def create_stocastic_dependence(frequency, false_ratio, vector_length):

    # Create boolean vector
    stocastic_dependence = np.ones(vector_length)  # , dtype=bool

    # Set false values at given intervals
    for i in range(0, vector_length, frequency):
        stocastic_dependence[i] = 0

    # Add stochastic noise
    noise = np.random.random(vector_length)
    stocastic_dependence[noise < false_ratio] = 0#False

    # Print boolean vector
   
    return stocastic_dependence


def plot_stocastic_dependence(vector):

    # Convert boolean values to numerical values
    vector = vector.astype(int)

    # Calculate power spectrum
    fft = np.fft.fft(vector)
    power_spectrum = np.abs(fft)**2


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot time series of boolean vector
    ax1.plot( vector)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Event occurence')
    ax1.set_title('Time Series')

    # Plot power spectrum of boolean vector
    ax2.plot(np.arange(len(vector))*frequency**2/vector_length, power_spectrum)  #
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power')
    ax2.set_title('Power Spectrum')

    plt.tight_layout()


def plot_traitTime_evol(trait_series, time_series):

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot time series of boolean vector
    ax1.plot(trait_series)
    ax1.set_xlabel('step')
    ax1.set_ylabel('knowledge')
    ax1.set_title('knowledge Series')

    # Plot power spectrum of boolean vector
    ax2.plot(time_series)  #
    ax2.set_xlabel('step')
    ax2.set_ylabel('Delta t')
    ax1.set_title('Time Series')

    plt.tight_layout()


def plot_multiple_traitTime_evol(ax1, ax2, trait_series, time_series, alpha, kThreshold, lw =0.3):

    # Plot time series of boolean vector
    ax1.plot(trait_series, color = 'orange', lw = lw, alpha = alpha)
    ax1.set_xlabel('step')
    ax1.set_ylabel('knowledge')
    ax1.set_title('knowledge Series')

    # Plot power spectrum of boolean vector
    ax2.plot(time_series,  color='blue', lw = lw, alpha=alpha)  #
    ax2.set_xlabel('step')
    ax2.set_ylabel('Delta t')
    ax1.set_title('Time Series')

    plt.tight_layout()

    # Print amplitude at frequency of interest
    #print(f"Amplitude at {signal_frequency} Hz: {np.abs(dft[freq_index])}")


def explore_frequency_range():

    a1 = np.logspace(0, 2, 3)
    a2 = np.arange(1, 2, 1)
    log_frequency_range = np.outer(a1, a2).flatten()
    print('frequencies', 500*time_step*log_frequency_range)

    for f in log_frequency_range:
        k_series_set, Dt_series_set, len_series_set = multiple_realizations(
            Nrealizations, time_step, k0,  kThreshold, int(f), false_ratio,  decay)

        plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, int(f))
        
    return log_frequency_range

def explore_noise_range():

    noise_range = np.arange(0, 1, 0.01)

    return noise_range


def multiple_realizations(Nrealizations, time_step, k0,  kThreshold, frequency, false_ratio,  decay):
    # Create figure with two subplots
    
    len_series_set = []
    k_series_set = []
    Dt_series_set = []
    for i in range(Nrealizations):
        if i%111 ==1 :print ('iiiii', i)
        stocastic_dependence = create_stocastic_dependence(
            frequency, false_ratio, vector_length)
        trait_series, time_series = trait_evol(
            stocastic_dependence, decay, time_step, k0, kThreshold)
        
        k_series_set.append(trait_series)
        Dt_series_set.append(time_series)
        len_series_set.append(len(trait_series))
    
    return k_series_set, Dt_series_set, len_series_set

    # Plot the histogram
    
    
def plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, frequency):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.hlines(y=kThreshold, xmin=0, xmax=vector_length,
               ls='--', linewidth=1.2, color='r')
    ax1.set_title('Frequency = ', frequency)
    noise_range = np.arange(0, 0.5, 0.5/Nrealizations)
    for i in range(Nrealizations):
        alpha = noise_range[i]
        plot_multiple_traitTime_evol(
            ax1, ax2, k_series_set[i], Dt_series_set[i], alpha, kThreshold)


    # Flatten the array into a 1D array
    data = np.array(len_series_set).flatten()

    ax3.hist(data, bins=25)
    ax3.set_xlabel('len of serie')
    ax3.set_ylabel('Frequency')


### parameters###
time_step = 0.01
frequency = int(500*time_step)
false_ratio = 0.66
vector_length = 1000
decay = frequency/2.3
k0 = 1 #initial knowledge
kThreshold = 1/10.*k0 
Nrealizations = 33


stocastic_dependence =  create_stocastic_dependence(frequency, false_ratio, vector_length)
trait_series, time_series = trait_evol(stocastic_dependence, decay, time_step, k0, kThreshold)
#print(trait_series)
plot_stocastic_dependence(stocastic_dependence)
plot_traitTime_evol(trait_series, time_series)

explore_frequency_range()
#k_series_set, Dt_series_set, len_series_set = multiple_realizations(
#    Nrealizations, time_step, k0,  kThreshold, frequency, false_ratio,  decay)
#plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set)

plt.show()








