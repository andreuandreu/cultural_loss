import string
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import unittest
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable


'''
what would be, eventually, a trait decay inside a knode, this code shall be eficient, fast, repeated

called "drifwood model" as it tries to model what is the threshold that the presence of a resouce/dependence
has to occur in order for the trait to remain "knowlwgelable" onside of the knode

'''
#numerical par
time_step = 0.5
Nrealizations = 333
vector_length = 25000


#model var
false_ratio = 0.44066  # 0.66
k0 = 1  # initial knowledge
kError = 1/10.*k0
decay = 0.001  # np.log(kError)/peride

root = './data/output/basic_decay'

periode = int(-time_step*np.log(kError)/decay)
max_period = int(83/time_step)#int(-time_step*np.log(kError)/decay)
min_period = 2 # do never go below 1 for ressolution issues. 
step_period = (max_period - min_period)/11.
periodes= np.arange(min_period, max_period, step_period)
print('peripperi', periodes)

max_falses = 0.6#0.66
min_falses = 0.01  
step_falses = (max_falses - min_falses)/11.
falses_ratios = np.arange(min_falses, max_falses, step_falses)
print('nininini', falses_ratios)
#visualization par
hist_bins = 18




class modelVar:
    def __init__(self, periode, false_ratio, decay, kError, k0):
        self.periode = periode
        self.periodes= periodes
        self.false_ratio = false_ratio
        self.falses_ratios = falses_ratios
        self.decay = decay
        self.kError = kError
        self.k0 = k0


class modelPar:
    def __init__(self, time_step, vector_length, Nrealizations):
        self.time_step = time_step
        self.vector_length = vector_length
        self.Nrealizations = Nrealizations

# Here's how you can create an instance of the Car class and call its display_info method:



def while_loop():
    threshold = 0.5
    k = [1]
    i = 0
    while i < 10 and k[i] > threshold:
        
        k.append( k[i-1] - 0.1)
        i = i + 1
        print(i, k)

def trait_evol( stocastic_dependence, var, par):
    
    kt = [k0]
    time_vec = [time_step]
    i = 0
    while i < len(stocastic_dependence)-1:
        
        t = time_step
        #if i > 988:
        #    print('iiiuuuu', i, t)
        while i < len(stocastic_dependence)-1 and stocastic_dependence[i] == 0 and kt[i] > var.kError:
            t = t + time_step
            i = i+1
            kt.append(kt[i-1]*np.exp(-t*var.decay))
            time_vec.append(t)
            #if i > 998:
            #    print('tttt it ended!', t, kt[i])
            
        if kt[i] > var.kError: 
            kt.append(k0)
            time_vec.append(par.time_step)
        else: 
            #print("end of loop at ", i, kt[i], time_vec[i])
            break

        i = i+1
    
    return kt , time_vec


def create_stocastic_dependence(var, par):

    # Create boolean vector
    stocastic_dependence = np.zeros(vector_length)  # , dtype=bool

    # Set false values at given intervals
    
    iminus = 0
    for i in range(0, par.vector_length, int(var.periode)):
        iminus  = i
        if np.random.random() > var.false_ratio:
            stocastic_dependence[i] = 1


    # Add stochastic noise
    #noise = np.random.random(par.vector_length)
    #stocastic_dependence[noise < var.false_ratio] = 0#False

    # Print boolean vector
   
    return stocastic_dependence


def plot_stocastic_dependence(vector, var, par):

    # Convert boolean values to numerical values
    vector = vector.astype(int)

    # Calculate power spectrum
    fft = np.fft.fft(vector)
    power_spectrum = np.abs(fft)**2


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot time series of boolean vector
    ax1.plot(np.arange(len(vector))*par.time_step, vector)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Recurrent event')
    ax1.set_title('recurrence series $T = $' + str(var.periode) + '$[t_s]$ $T_{\epsilon} = $' + str(var.false_ratio) +
                  ' $k_{\epsilon} = $' + str(var.kError) + ' $d = $' + "{:.2}".format(var.decay))

    # Plot power spectrum of boolean vector
    x_scale = np.arange(2*par.vector_length/var.periode) * \
        var.periode**2/par.vector_length
    ax2.plot(x_scale, power_spectrum[0:len(x_scale)])  #
    #ax2.plot(np.arange(len(vector))*par.time_step, power_spectrum)
    ax2.set_xlabel('periode')
    ax2.set_ylabel('Power')
    ax2.set_title('Power Spectrum')

    #plt.tight_layout()


def plot_traitTime_evol(trait_series, time_series):

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '$[t_s]$ $T_{\epsilon}= $' + str(var.false_ratio) +
                  ' $k_{\epsilon} = $' + str(var.kError) + ' $d = $' + "{:.2}".format(var.decay))

    # Plot time series of boolean vector
    ax1.plot(trait_series)
    ax1.set_xlabel('step')
    ax1.set_ylabel('knowledge')

    # Plot power spectrum of boolean vector
    ax2.plot(time_series)  #
    ax2.set_xlabel('step')
    ax2.set_ylabel('Delta t')


    plt.tight_layout()


def plot_multiple_traitTime_evol(ax1, ax2, trait_series, time_series, var, alpha, lw =0.3):

    # Plot time series of boolean vector
    ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '$[t_s]$ $T_{\epsilon}= $' + str(var.false_ratio) +
                  ' $k_{\epsilon} = $' + str(var.kError) + ' $d = $' + "{:.2}".format(var.decay))
    ax1.plot(trait_series, color = 'orange', lw = lw, alpha = alpha)
    ax1.set_xlabel('step')
    ax1.set_ylabel('knowledge')

    # Plot power spectrum of boolean vector
    ax2.plot(time_series,  color='blue', lw = lw, alpha=alpha)  #
    ax2.set_xlabel('step')
    ax2.set_ylabel('Delta t')
    

    #plt.tight_layout()

    # Print amplitude at periode of interest
    #print(f"Amplitude at {signal_periode} Hz: {np.abs(dft[freq_index])}")


def explore_periode_range(var, par):

    len_data_series = []

    for T in var.periodes:
        var.periode = int(T)
        print('pepeprpepreprpe', var.periode, 'nonono', var.false_ratio )
        k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
            multiple_noiseRealizations(var, par)

        len_data_series.append(np.array(len_series_set).flatten())

        #plot_stocastic_dependence(one_stocastic_dependence, var, par)
        #plot_traitTime_evol(one_trait_series, one_time_series)
        #plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, hist_bins)
        
        name = root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(T) + '_Te=' +\
            "{:.2f}".format(var.false_ratio) + '_ke=' + str(var.kError) + \
            '_d=' + "{:.2}".format(var.decay)+'.npy'
        np.save(name, np.array(len_series_set).flatten())
    
    return len_data_series

def explore_noise_range(var, par):

    len_data_series = []

    for n in var.falses_ratios:
        print('nnnnn', n)
        var.false_ratio = n

        explore_periode_range(var, par)
        #k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
        #    multiple_noiseRealizations(var, par)

        #len_data_series.append(np.array(len_series_set).flatten())

        #plot_stocastic_dependence(one_stocastic_dependence, var, par)
        #plot_traitTime_evol(one_trait_series, one_time_series)
        #plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, hist_bins)

        #name = root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(var.periode) + '_Te=' +\
        #    "{:.2f}".format(n) + '_ke=' + str(var.kError) + \
        #    '_d=' + "{:.2}".format(var.decay)+'.npy'
        #np.save(name, np.array(len_series_set).flatten())

    return len_data_series


def multiple_noiseRealizations(var, par):

    len_series_set = []
    k_series_set = []
    Dt_series_set = []
    for i in range(par.Nrealizations):
        if i%111 ==1 :print ('iiiii', i)
        stocastic_dependence = create_stocastic_dependence(
            var, par)
        trait_series, time_series = trait_evol(
            stocastic_dependence, var, par)
        
        k_series_set.append(trait_series)
        Dt_series_set.append(time_series)
        len_series_set.append(len(trait_series))
    
    return k_series_set, Dt_series_set, len_series_set, stocastic_dependence, trait_series, time_series

    
def plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, par, hist_bins):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    

    ax1.hlines(y=kError, xmin=0, xmax=vector_length,
               ls='--', linewidth=1.2, color='r')
   
    noise_range = np.arange(0, 0.5, 0.5/Nrealizations)
    for i in range(Nrealizations):
        alpha = noise_range[i]
        plot_multiple_traitTime_evol(
            ax1, ax2, k_series_set[i], Dt_series_set[i], par, alpha)


    # Flatten the array into a 1D array
    data = np.array(len_series_set).flatten()
    #print('dadada', data)
    ax3.hist(data, bins=hist_bins)
    ax3.set_xlabel('len of serie')
    ax3.set_ylabel('periode')


def plot_Period_decay_dep(var, par):
    fig, ax = plt.subplots()

    a1 = 1/np.logspace(0, 3, 4)
    a2 = np.arange(0.001, -np.log(var.kError), 1)
    log_decay_range = np.outer(a1, a2).flatten()*par.time_step
    print('decays', log_decay_range*par.time_step)
    Tmax = - np.log(var.kError)/log_decay_range
    #print('Tmaxssss', Tmax)
    ax.scatter(log_decay_range, Tmax )
    ax.set_xscale("log")
    ax.set_yscale("log")

def plot_survival_martrix(rows, cols):

    survival_rate = np.empty([rows, cols])
    for i in range(rows):
        for j in range(cols):
            print('noise', "{:.2f}".format(var.falses_ratios[i]), 'per', "{:.2f}".format(var.periodes[j]))
            name = root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(var.periodes[j]) + '_Te=' +\
                "{:.2f}".format(var.falses_ratios[i]) + '_ke=' + str(var.kError) + \
                '_d=' + "{:.2}".format(var.decay)+'.npy'
            dataset_len_series = np.load(name)
            # print('ufufufuf', dataset_len_series)
            survivors = len(np.where(dataset_len_series >
                            par.vector_length-10)[0])
            survival_rate[i][j] = survivors/par.Nrealizations
            print('sisisiusususu', survivors/par.Nrealizations)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(survival_rate, cmap='OrRd')  # extent = extent, 'bone'

    labelx = []
    for e in var.periodes:
        labelx.append("{:.0f}".format(e*par.time_step))
    labely = []
    for e in var.falses_ratios:
        labely.append("{:.2f}".format(e))

    ax.set(xticks=np.arange(len(var.periodes)), xticklabels=labelx,
           yticks=np.arange(len(var.falses_ratios)), yticklabels=labely)
    ax.set_xlabel('$T$')
    ax.set_ylabel('$T_{\epsilon}$')

    fig.colorbar(im, cax=cax, orientation='vertical')   
    
 

def multiplot_NxM(rows, cols, par, var, hist_bins):


    # create a figure and set the size
    fig1, axs = plt.subplots(rows, cols, sharey=True, subplot_kw=dict(
        frameon=False))  # sharex=True, sharey=True
    

    #axs.set_xlabel('len of serie')
    #axs.set_ylabel('periode')

    l = 0
    for i in range(rows):
        for j in range(cols):
            if len(var.periodes) == l:
                break
            else:
                name = root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(var.periodes[j]) + '_Te=' +\
                    "{:.2f}".format(var.falses_ratios[i]) + '_ke=' + str(var.kError) + \
                    '_d=' + "{:.2}".format(var.decay)+'.npy'
                dataset_len_series = np.load(name)
                #print('ufufufuf', dataset_len_series)
                
                #axs.set_title('T=' + "{:.2f}".format(var.periodes[j]))
                axs[i][j].hist(dataset_len_series, bins=hist_bins)
            l += 1

    #ax2.set_title('survival rate')
    #ax2.plot(var.periodes, survival_rate)
    
    # adjust spacing between subplots
    #fig.tight_layout()



var = modelVar(periode, false_ratio, decay, kError, k0)
par = modelPar(time_step, vector_length, Nrealizations)
plot_Period_decay_dep(var, par)

#stocastic_dependence =  create_stocastic_dependence(var, par)
#trait_series, time_series = trait_evol(stocastic_dependence, var, par)
#print(trait_series)


#len_data_series = explore_periode_range(var, par)
#len_data_series =  explore_noise_range(var, par)
m = len(var.falses_ratios)
n = len(var.periodes)
#multiplot_NxM(m, n, par, var, hist_bins)
plot_survival_martrix(m, n)
#k_series_set, Dt_series_set, len_series_set = multiple_realizations(
#    Nrealizations, time_step, k0,  kError, periode, false_ratio,  decay)
#plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set)

plt.show()








