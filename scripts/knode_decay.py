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
import os
import sys

#python scripts/knode_decay.py 'th_1-e(-Tts)-1' 44
'''
what would be, eventually, a trait halfLife inside a knode, this code shall be eficient, fast, repeated

called "drifwood model" as it tries to model what is the threshold that the presence of a resouce/dependence
has to occur in order for the trait to remain "knowlwgelable" onside of the knode

'''
#numerical par
time_step = 0.5
Nrealizations = 333
vector_length = 2000


#model var
output_dir = './data/output/'
root = sys.argv[1] # 'th_pop' # 'noisTpc_th-e-(Tts)-1'  # 'noisTpc_th-2t12'
plots_dir = './plots/embers/'
plots_survivalMat = 'fig_survivalMatrix/'


periode = 10/time_step#int(-time_step*np.log(kError)/halfLife)
max_period =int(83/time_step)#int(-time_step*np.log(kError)/halfLife)
min_period =2 # do never go below 1 for ressolution issues. 
step_period =(max_period - min_period)/44.
periodes= np.arange(min_period, max_period, step_period)
print('peripperi', periodes)

halfLife = float(sys.argv[2])#88#np.log(2)/periode  # 0.05  #
max_halfLife = (75/time_step)  # int(-time_step*np.log(kError)/halfLife)
min_halfLife = 8   # do never go below 1 for ressolution issues.
step_halfLife = (max_halfLife - min_halfLife)/44
halfLifes = np.arange(min_halfLife, max_halfLife, step_halfLife)
print('dededededekay', halfLifes)


k0 = 1  # initial knowledge
kError = k0 * np.exp(-halfLife/(periode*time_step))  # 1/10.*k0
min_kError = 0.01
max_kError = 0.95
step_kError = (max_kError - min_kError)/10
kErrors = np.append(np.array([min_kError]), np.arange(min_kError, max_kError, step_kError))
print('kekekekkee', kErrors)

false_ratio = 0.11
max_falses = 0.02#0.6#0.66
min_falses = 0.01  
step_falses = (max_falses - min_falses)/1.
falses_ratios = np.arange(min_falses, max_falses, step_falses)
print('falsfalsfalse', falses_ratios)

noiseLevel = 2.0
max_noises = 5.1#0.0002  # 0.6#0.66
min_noises = 0.1#0.0001
step_noises = (max_noises - min_noises)/11.
noiseLevels = np.arange(min_noises, max_noises, step_noises)
print('nisnosisnoise', noiseLevels)

population = 50
max_pop = 5.1  # 0.0002  # 0.6#0.66
min_pop = 0.1  # 0.0001
step_pop = (max_pop - min_pop)/11.
populations = np.arange(min_pop, max_pop, step_pop)
print('popopopopopo', populations)


# visualization par
hist_bins = 18



class modelVar:
    def __init__(self, periode, false_ratio, halfLife, kError, k0):
        self.periode = periode
        self.periodes= periodes

        self.false_ratio = false_ratio
        self.falses_ratios = falses_ratios

        self.noiseLevel = noiseLevel
        self.noiseLevels = noiseLevels

        self.population = population
        self.populations = populations

        self.halfLife = halfLife
        self.halfLifes = halfLifes

        self.kError = kError
        self.kErrors = kErrors
        self.k0 = k0


class modelPar:
    def __init__(self, time_step, vector_length, Nrealizations):
        self.time_step = time_step
        self.vector_length = vector_length
        self.Nrealizations = Nrealizations
        self.output_dir = output_dir
        self.root = root
        self.plots_dir = plots_dir
        self.plots_survivalMat = plots_survivalMat


def which_threshlod(var, par):

    if 'th-2t12' in par.root:
        return k0 * 2**(-2*var.halfLife/(var.periode ))
    elif 'th_pop' in par.root:
        return k0 * int(1/var.population)
    elif 'th-t12' in par.root:
        return k0 * 2**(-var.halfLife/(var.periode))
    elif 'th-tau' in par.root:
        return k0 * np.exp(-var.halfLife*np.log(2)/(var.periode ))
    elif 'th-2tau' in par.root:
        return k0 * np.exp(-2*var.halfLife*np.log(2)/(var.periode ))
    elif 'th-e(-Tts)-1' in par.root:
        return k0 * np.exp(-1/(var.periode*par.time_step ))
    elif 'th-e(-t12ts)-1' in par.root:
        return k0 * np.exp(-1/(var.halfLife*par.time_step))
    elif 'th_1-e(-Tts)-1' in par.root:
        return k0 * (1- np.exp(-1/(var.periode*par.time_step)))
    elif 'th_1-e(-ts)-1' in par.root:
        return k0 * (1 - np.exp(-1/(var.halfLife*0.5)))
    elif 'th_1-e(-t12T)' in par.root:
        return k0 * ( 1- np.exp(-var.halfLife/(var.periode)))
    elif 'th_e(-t12T)' in par.root:
        return k0 * (np.exp(-var.halfLife/(var.periode)))
    elif 'th_e(-t12T)' in par.root:
        return k0 * (1-np.exp(-var.halfLife/(var.periode)))
    elif 'th_2(-Tt12)' in par.root:
        return k0 * (2**(-var.periode/(var.halfLife)))
    elif 'th_1-2(-Tt12)' in par.root:
        return k0 * (1-2**(-var.periode/(var.halfLife)))
    elif 'th_1-2(-t12ts)' in par.root:
        return k0 * (1-2**(-1/(var.halfLife*par.time_step)))
    elif 'th_05' in par.root:
        return k0 * 0.5
    elif 'th_01' in par.root:
        return k0 * 0.1
    else:
        print('NON OF THE OPTIONS GIVEN WORKS!! CHECK THE THRESHOLD FLAGS')
        quit()

def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        # pairwise unequal (string safe)
        y = ia[1:] != ia[:-1]
        # must include last element posi
        i = np.append(np.where(y), n - 1)
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def trait_evol( stocastic_dependence, var, par):
    
    kt = [k0]
    time_vec = [time_step]
    i = 0
    threshold = which_threshlod(var, par)
    while i < len(stocastic_dependence)-1:
        
        t = time_step
        #if i > 988:
        #    print('iiiuuuu', i, t)
        while i < len(stocastic_dependence)-1 and stocastic_dependence[i] == 0 and kt[i] > threshold: 
            t = t + time_step
            i = i+1
            #kt.append(kt[i-1]*np.exp(-t*var.halfLife))
            #kt.append(k0*np.exp(-t/(par.time_step*var.halfLife*np.log(2))))
            k = k0*np.exp(-t/(par.time_step*var.halfLife*np.log(2)))
            discrete_k = int(var.population * k)/var.population
            kt.append(discrete_k)
            #kt.append(k0*2**(-t/(par.time_step*var.halfLife)))
        
            time_vec.append(t)
            #if i > 998:
            #    print('tttt it ended!', t, kt[i])
            
        if kt[i] > threshold:  # var.kError:
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


def create_noisy_period_series(var, par):

    # Create boolean vector
    stocastic_dependence = np.zeros(par.vector_length)  # , dtype=bool

    indexes = np.arange(0, par.vector_length, var.periode)
    #noise = np.random.random(len(indexes))*var.noiseLevel
    noise = np.random.normal(
        loc=0, scale=var.noiseLevel*var.periode, size=len(indexes))

    #print('nonononono', var.periode, len(indexes),
    #      len(noise), par.vector_length)
    #index = np.where((noise.astype('int') + indexes)  % var.periode == 0)

    sum = noise.astype('int') + indexes.astype('int')
    index = np.where((sum >= 0) & (sum < par.vector_length))
    
    #print('ufff', var.periode, (
    # noise.astype('int')[:10] + indexes[:10]))
    #print('inininin', index[:10])
    stocastic_dependence[sum[index]] = 1.


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
    per_in_yr = var.periode*par.time_step
    ax1.set_title('recurrence series $T = $' + "{:.1f}".format(per_in_yr) + \
                   '[yr] $T_{\epsilon} = $' + "{:.2}".format(var.false_ratio) + \
                  ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' $t12 =$' + \
                  "{:.1f}".format(var.halfLife*par.time_step) + '[yr]')

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

    ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '[yr] $T_{\epsilon}= $' + "{:.2f}".format(var.noiseLevel) +
                  ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' $t_{1/2} =$' + "{:.1f}".format(var.halfLife) + '[yr]')

    # Plot time series of boolean vector
    ax1.plot(np.arange(len(trait_series))*par.time_step, trait_series)
    #ax1.set_xlabel('time')
    ax1.set_ylabel('knowledge')

    # Plot power spectrum of boolean vector
    ax2.plot(np.arange(len(trait_series))*par.time_step, time_series)  #
    ax2.set_xlabel('time')
    ax2.set_ylabel('Delta t')


    plt.tight_layout()


def plot_multiple_traitTime_evol(ax1, ax2, trait_series, time_series, var, par, alpha, lw =0.3):

    # Plot time series of boolean vector
    ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '$[t_s]$ $T_{\epsilon}= $' + "{:.2f}".format(var.noiseLevel) +
                  ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' $t12 =$' + "{:.1f}".format(var.halfLife))
    ax1.plot(np.arange(len(trait_series))*par.time_step,
              trait_series, color='orange', lw=lw, alpha=alpha)
    #ax1.set_xlabel('step')
    ax1.set_ylabel('knowledge')

    # Plot power spectrum of boolean vector
    ax2.plot(np.arange(len(trait_series))*par.time_step, time_series,  color='blue', lw=lw, alpha=alpha)  #
    ax2.set_xlabel('time')
    ax2.set_ylabel('Delta t')
    

    #plt.tight_layout()

    # Print amplitude at periode of interest
    #print(f"Amplitude at {signal_periode} Hz: {np.abs(dft[freq_index])}")


def explore_oneVar_range(var, par, varName, varRange):

    len_data_series = []

    for v in varRange:
        name = file_name_n_varValue(varName, v)
        print('name var', varName, 'val', v )
        k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
            multiple_noiseRealizations(var, par)
        len_data_series.append(np.array(len_series_set).flatten())
        #plot_stocastic_dependence(one_stocastic_dependence, var, par)
        #plot_traitTime_evol(one_trait_series, one_time_series)
        #plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, hist_bins)
        np.save(name, np.array(len_series_set).flatten())

    return len_data_series


def explore_twoVar_ranges(var, par, varNameX, varRangeX, varNameY, varRangeY ):

    for v in varRangeY:
        name = file_name_n_varValue(varNameY, v)
        print('name var', varNameY, 'val', v)
        explore_oneVar_range(var, par, varNameX, varRangeX)

    
def explore_periode_range(var, par):

    len_data_series = []

    for T in var.periodes[2:4]:
        var.periode = int(T)
        var.kError = which_threshlod(var, par)
        print('pepeprpepreprpe', var.periode, 'nonono', var.noiseLevel )
        k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
            multiple_noiseRealizations(var, par)

        len_data_series.append(np.array(len_series_set).flatten())

        plot_stocastic_dependence(one_stocastic_dependence, var, par)
        plot_traitTime_evol(one_trait_series, one_time_series)
        plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, par, hist_bins)
        
        name = file_name_n_varValue('periode',T)  # '  # str(var.kError) +
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
        #plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, par, hist_bins)

        #np.save(name, np.array(len_series_set).flatten())

    return len_data_series



def explore_halfLife_range(var, par):


    for n in var.halfLifes:
        print('ddddddd', n)
        var.halfLife = n

        explore_periode_range(var, par)

    return 


def multiple_noiseRealizations(var, par):

    len_series_set = []
    k_series_set = []
    Dt_series_set = []
    for i in range(par.Nrealizations):
        if i%111 ==1 :print ('iiiii', i)
        #stocastic_dependence = create_stocastic_dependence( var, par)
        stocastic_dependence = create_noisy_period_series(var, par)
        trait_series, time_series = trait_evol(
            stocastic_dependence, var, par)
        
        k_series_set.append(trait_series)
        Dt_series_set.append(time_series)
        len_series_set.append(len(trait_series))
    
    return k_series_set, Dt_series_set, len_series_set, stocastic_dependence, trait_series, time_series

    
def plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, par, hist_bins):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    
    var.kError = which_threshlod(var, par)
    ax1.hlines(y=var.kError, xmin=0, xmax=vector_length*par.time_step,
               ls='--', linewidth=1.2, color='r')
   
    noise_range = np.arange(0, 0.5, 0.5/Nrealizations)
    for i in range(Nrealizations):
        alpha = noise_range[i]
        plot_multiple_traitTime_evol(
            ax1, ax2, k_series_set[i], Dt_series_set[i], var, par, alpha)


    # Flatten the array into a 1D array
    data = np.array(len_series_set).flatten()
    #print('dadada', data)
    ax3.hist(data, bins=hist_bins)
    ax3.set_xlabel('len of serie')
    ax3.set_ylabel('periode')


def plot_Period_halfLife_dep(var, par):
    fig, ax = plt.subplots()

    a1 = 1/np.logspace(0, 3, 4)
    a2 = np.arange(0.001, -np.log(var.kError), 1)
    log_halfLife_range = np.outer(a1, a2).flatten()*par.time_step
    print('halfLifes', log_halfLife_range*par.time_step)
    Tmax = - np.log(var.kError)/log_halfLife_range
    #print('Tmaxssss', Tmax)
    ax.scatter(log_halfLife_range, Tmax )
    ax.set_xscale("log")
    ax.set_yscale("log")


def file_name_n_varValue(to_modify, value):

    if to_modify == 'periode':
        var.periode = value
        
    elif to_modify == 'halfLife':
        var.halfLife = value
        
    elif to_modify == 'false_ratio':
        var.false_ratio = value

    elif to_modify == 'noiseLevel':
        var.noiseLevel = value
        
    elif to_modify == 'kError':
        var.kError = value
        
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, halfLife, false_ratio, kError')
        quit()

    path = output_dir + root + '_ts='+str(par.time_step) + '_L=' + str(par.vector_length) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    name = path +'T=' + "{:.2f}".format(var.periode) + '_Te=' +\
        "{:.2f}".format(var.noiseLevel) + '_t12=' + "{:.1f}".format(var.halfLife)+'.npy'  # + str(var.kError)
    
    return name


def name_survival_fig(to_modify, root_fig):
    
    if 'periode' in to_modify:
        periode_seg = '_Tran-'+"{:.2f}".format(var.periodes[0])+'-'+"{:.2f}".format(var.periodes[-1])
    else:
        periode_seg = '_T-' + "{:.2f}".format(var.periode)

    if 'halfLife' in periode_seg:
        halfLife_seg = '_t12ran-'+"{:.1f}".format(var.halfLifes[0])+'-'+"{:.1f}".format(var.halfLifes[-1])
    else:
        halfLife_seg = '_t12-' + "{:.1f}".format(var.halfLife)

    if 'false_ratio' in to_modify:
        false_ratio_seg = '_FRran-'+"{:.2f}".format(var.falses_ratios[0])+' -'+"{:.2f}".format(var.falses_ratios[-1])
    else:
        false_ratio_seg = '_FR-' + "{:.2f}".format(var.false_ratio)

    if 'noiseLevel' in to_modify:
        noiseLevel_seg = '_TNran-'+"{:.2f}".format(var.noiseLevels[0])+'-'+"{:.2f}".format(var.noiseLevels[-1])
    else:
        noiseLevel_seg = '_TN-' + "{:.2f}".format(var.noiseLevel)

    if  'kError' in to_modify:
        kError_seg = '_kEran-'+"{:.2f}".format(var.kErrors[0])+' -'+"{:.2f}".format(var.kErrors[-1])
    else:
        kError_seg = '_kE-' + "{:.2f}".format(var.kError)

    path = plots_dir + root_fig + par.root + '_ts=' + \
        str(par.time_step) + '_L='+str(par.vector_length) + \
        '_N=' + str(par.Nrealizations) + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    name_fig = path + 'fig' + periode_seg + noiseLevel_seg + halfLife_seg + '.png'  # + kError_seg  +

    return name_fig 


def var_tagAndLabels(varName, values):

    labels = []

    if varName == 'periode':
        tag = '$T$[yr]'
        for e in values:
            labels.append("{:.0f}".format(e*par.time_step))

    elif varName == 'halfLife':
        tag = '$t_{1/2}$[yr]'
        #tag = '$\lambda$[yr$^-1$]'
        for e in values:
            labels.append("{:.2f}".format(par.time_step*e))

    elif varName == 'false_ratio':
        tag='$T_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2}".format(e))

    elif varName == 'noiseLevel':
        tag = '$T_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2}".format(e))

    elif varName == 'kError':
        tag = '$k_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2f}".format(e))
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, halfLife, false_ratio, kError')
        quit()

    return tag, labels

def set_title_mat(varNameX, varNameY, num, maxNum):

    if varNameX == 'periode' and varNameY == 'noiseLevel':
        if num == 0:
            return '$t_{1/2}$ = ' + "{:4.0f}".format(var.halfLife*par.time_step)
        elif num == maxNum-1:
            return "{:4.0f}".format(var.halfLife*par.time_step) + '[yr]'
        else:
            return "{:4.0f}".format(var.halfLife*par.time_step)
    elif varNameX == 'periode' and varNameY == 'halfLife':
        return '$T_{\epsilon}$ = ' + "{:2.1f}".format(100*var.noiseLevel) + '[%]'
    elif varNameX == 'noiseLevel' and varNameY == 'halfLife':
        if num == 0:
            return '$T$ = ' + "{:d}".format(var.periode)
        elif num == maxNum-1:
            return "{:d}".format(var.periode) + '[yr]'
        else: 
            return "{:d}".format(var.periode)
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, halfLife, false_ratio, kError')
        quit()
    

def plot_survival_martrix(varNameY, valuesY, varNameX, valuesX):

    fig, ax = plt.subplots()
    survival_rate = np.empty([len(valuesY), len(valuesX)])
    for i, valY in enumerate(valuesY):
        nameY = file_name_n_varValue(varNameY, valY)
        for j, valX  in enumerate(valuesX):
            #print('noise', "{:.2f}".format(var.false_ratio), 'per', "{:.2f}".format(var.periode))

            nameX = file_name_n_varValue(varNameX, valX)
            dataset_len_series = np.load(nameX)
            # print('ufufufuf', dataset_len_series)
            survivors = len(np.where(dataset_len_series >
                            par.vector_length-10)[0])
            survival_rate[i][j] = survivors/par.Nrealizations
            #print('sisisiusususu', survivors/par.Nrealizations)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    x, y = np.meshgrid(var.periodes, var.halfLifes)
    # extent = extent, 'bone'
    im = ax.imshow(survival_rate,   extent=[x.min(), x.max(), y.max(), y.min()], cmap='OrRd')
    ax.plot(var.periodes, var.halfLifes/np.log(var.noiseLevel))

    #for e in var.noiseLevels:
    #    ax.plot(var.periodes, var.halfLifes*e/np.log(2))
    #ax.plot(var.periode, var.halfLife)
    #ax.plot(var.periode, var.halfLife*var.noiseLevel)
    tagX, labelsX = var_tagAndLabels(varNameX, valuesX)
    tagY, labelsY = var_tagAndLabels(varNameY, valuesY)

    #ax.set(xticks=np.arange(len(valuesX)), xticklabels=labelsX,
    #       yticks=np.arange(len(valuesY)), yticklabels=labelsY)
    
    #title = '$t_{1/2}$ = ' + "{:4.0f}".format(var.halfLife*par.time_step) + '[yr]'
    title = '$T_{\epsilon}$ = ' + \
        "{:3.1f}".format(100*var.noiseLevel) + '[%]'
    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)
    ax.set_title(title)

    fig.colorbar(im, cax=cax, orientation='vertical')   

    to_modify = varNameX + '_' + varNameY
    name_fig = name_survival_fig(to_modify, par.plots_survivalMat)

    
    #plot(ax1, V.real)

    #ax.plot(par.vector_length *
    #        0.005/(var.periodes*par.time_step)**2)

    print('nananannana', name_fig)
    #plt.savefig(name_fig, bbox_inches='tight')
 

def multiplot_NxM(rows, cols, par, var, varName, hist_bins):

    # create a figure and set the size
    fig1, axs = plt.subplots(rows, cols, sharey=True, subplot_kw=dict(
        frameon=False))  # sharex=True, sharey=True

    # axs.set_xlabel('len of serie')
    # axs.set_ylabel('periode')

    l = 0

    for i in range(rows):
        for j in range(cols):
            if len(var.periodes) == l:
                break
            else:
                name = file_name_n_varValue(varName, var.peridodes[j])
                dataset_len_series = np.load(name)
                # print('ufufufuf', dataset_len_series)

                # axs.set_title('T=' + "{:.2f}".format(var.periodes[j]))
                axs[i][j].hist(dataset_len_series, bins=hist_bins)
            l += 1


def multiplot_survivals(varNameY, valuesY, varNameX, valuesX, varName, varArr, rows=1):

    # create a figure and set the size
    cols = len(varArr)
    #fig, axs = plt.subplots(rows, cols, sharey=True, subplot_kw=dict(
    #    frameon=False))  # sharex=True, sharey=True
    fig, axs = plt.subplots(rows, cols)
    
    tagX, labelsX = var_tagAndLabels(varNameX, valuesX)
    tagY, labelsY = var_tagAndLabels(varNameY, valuesY)

    mossaic_keys = [ ['0', '1', '2', '3', '4', '5' ]]

    hight = 4
    fig, axs = plt.subplot_mosaic(
        mossaic_keys,
        sharex=True,
        sharey=True,
        figsize=(12.1, hight),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )


    labelsX_short  = []
    intervalX = 5
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])


    #for i in range(rows):
    for j in range(cols):
        #name = file_name_n_varValue(varName, varArr[j])
        if varName == 'halfLife':
            var.halfLife = varArr[j]
        
        survival_rate = a_survival_martrix(
            varNameY, valuesY, varNameX, valuesX)
        im = axs[str(j)].imshow(survival_rate, cmap='OrRd')
        if j == 0:
            axs[str(j)].set_ylabel(tagY)
            axs[str(j)].set(xticks=np.arange(0, len(var.halfLifes), intervalX), xticklabels=labelsX_short,
                   yticks=np.arange(len(valuesY)), yticklabels=labelsY)
        #if j > 0:
            #axs[str(j)] = axs[j-1].twiny()
            #axs[j].set(xticks=np.arange(len(valuesX)), xticklabels=labelsX)
        title = set_title_mat(varNameX, varNameY, j, cols)
        axs[str(j)].set_xlabel(tagX)
        axs[str(j)].set_title(title)

    im = axs[str(cols-1)].imshow(survival_rate, cmap = 'OrRd')
    divider=make_axes_locatable(axs[str(cols-1)])
    cax = divider.append_axes('right', size='0%', pad=0.0)
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    #to_modify = varNameX + '_' + varNameY
    #name_fig = name_survival_fig(to_modify, par.plots_survivalMat)

 

        

def a_survival_martrix(varNameY, valuesY, varNameX, valuesX):

    survival_rate = np.empty([len(valuesY), len(valuesX)])
    for i, valY in enumerate(valuesY):
        nameY = file_name_n_varValue(varNameY, valY)
        for j, valX in enumerate(valuesX):

            nameX = file_name_n_varValue(varNameX, valX)
            dataset_len_series = np.load(nameX)
            survivors = len(np.where(dataset_len_series >
                            par.vector_length-10)[0])
            survival_rate[i][j] = survivors/par.Nrealizations

    survival_rate[-1][-1] = 0
    return survival_rate
    # plt.savefig(name_fig, bbox_inches='tight')


def max_death_interval(var, par):

    n_max_mat = []
    for e in var.kErrors:
        n_max_row = []
        for l in var.halfLifes:
            n_max_row.append(par.time_step *np.log(k0/e)/l)
        n_max_mat.append(n_max_row)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    n_max_mat = np.array(n_max_mat)
    aux = np.where(n_max_mat > 86)
    n_max_mat[aux] = 0
    aux = np.where(n_max_mat < 2)
    n_max_mat[aux] = 86
    
    #im = ax.imshow(n_max_mat, cmap='bone')  # extent = extent, 'bone'
    x_tic_labels = np.log(2)*par.time_step/var.halfLifes
    im = ax.contourf(x_tic_labels, var.kErrors, n_max_mat,
                     extend="both", cmap='bone')

    tagX, labelsX = var_tagAndLabels('halfLife', var.halfLifes)
    tagY, labelsY = var_tagAndLabels('kError', var.kErrors)


    labelsX_short  = []
    intervalX = 20
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])

    labelsY_short = []
    intervalY = 10
    for i in range(len(labelsY)):
        if i % intervalY == 0:
            labelsY_short.append(labelsY[i])

    #ax.set(xticks=np.arange(0, len(var.halfLifes), intervalX), xticklabels=labelsX_short,
    #       yticks=np.arange(0, len(var.kErrors), intervalY), yticklabels=labelsY_short)

    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)


    fig.colorbar(im, cax=cax, orientation='vertical')



    return n_max_mat


def max_death_interval(var, par):

    n_max_mat = []
    for e in var.kErrors:
        n_max_row = []
        for l in var.halfLifes:
            n_max_row.append(par.time_step * np.log(k0/e)/l)
        n_max_mat.append(n_max_row)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    n_max_mat = np.array(n_max_mat)
    aux = np.where(n_max_mat > 86)
    n_max_mat[aux] = 0
    aux = np.where(n_max_mat < 2)
    n_max_mat[aux] = 86

    # im = ax.imshow(n_max_mat, cmap='bone')  # extent = extent, 'bone'
    x_tic_labels = np.log(2)*par.time_step/var.halfLifes
    im = ax.contourf(x_tic_labels, var.kErrors, n_max_mat,
                     extend="both", cmap='bone')

    tagX, labelsX = var_tagAndLabels('halfLife', var.halfLifes)
    tagY, labelsY = var_tagAndLabels('kError', var.kErrors)

    labelsX_short = []
    intervalX = 20
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])

    labelsY_short = []
    intervalY = 10
    for i in range(len(labelsY)):
        if i % intervalY == 0:
            labelsY_short.append(labelsY[i])

    # ax.set(xticks=np.arange(0, len(var.halfLifes), intervalX), xticklabels=labelsX_short,
    #       yticks=np.arange(0, len(var.kErrors), intervalY), yticklabels=labelsY_short)

    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)

    fig.colorbar(im, cax=cax, orientation='vertical')

    return n_max_mat



def plot_threshold_function(var, par):

    threshold_mat = []
    for h in var.halfLifes:
        threshold_row = []
        var.halfLife = h
        for l in var.periodes:
            var.periode = l
            threshold = which_threshlod(var, par)
            threshold_row.append(threshold)
        threshold_mat.append(threshold_row)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #n_max_mat = np.array(threshold_mat)
    #aux = np.where(threshold_mat > 86)
    #n_max_mat[aux] = 0
    #aux = np.where(threshold_mat < 2)
    #n_max_mat[aux] = 86

    # im = ax.imshow(n_max_mat, cmap='bone')  # extent = extent, 'bone'
    im = ax.contourf(var.periodes, var.halfLifes, threshold_mat,
                     extend="both", cmap='bone')

    tagX, labelsX = var_tagAndLabels('periode', var.periodes)
    tagY, labelsY = var_tagAndLabels('halfLife', var.halfLifes)

    #ax.set(xticks=np.arange(len(var.periodes)), xticklabels=labelsX,
    #       yticks=np.arange(len(var.halfLifes)), yticklabels=labelsY)

    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)

    fig.colorbar(im, cax=cax, orientation='vertical')

    return threshold_mat
 

def plot_threshold_functions_halfLife_periode():
    fig, ax = plt.subplots()

    threshold12 = np.exp(-halfLife/(periodes*time_step))
    threshold2 = np.exp(-2*halfLife/(periodes*time_step))
    threshold05 = np.exp(-halfLife/(2*periodes*time_step))
    threshold22 = 2**(-2*halfLife/(periodes*time_step))
    threshold14 = np.exp(-np.log(2)*halfLife/(periodes*time_step))
    threshold214 = np.exp(-2*np.log(2)*halfLife/(periodes*time_step))

    ax.plot(periodes, threshold12, c = 'b')
    ax.plot(periodes, threshold2, c = 'g')
    ax.plot(periodes, threshold05, c = 'r')
    ax.plot(periodes, threshold22, c='m')
    ax.plot(periodes, threshold14, c='y')


def plot_threshold_functions_preiode(var, par):
    fig, ax = plt.subplots()

    threshold1 = np.exp(-1/(var.periodes))
    threshold2 = np.exp(-1/(var.periodes*par.time_step))
    threshold05 = np.exp(-par.time_step/(var.periodes))

    #ax.plot(periodes, threshold1, c='y', label ='exp(-1/T)')
    #ax.plot(periodes, threshold2, c='g',  label='exp(-1/(T*ts))')
    #ax.plot(periodes, threshold05, c='r', label='exp(-ts/T)')

    #ax.plot(periodes, 1-threshold1, c='m', label='1-exp(-1/T)')
    ax.plot(periodes, 1-threshold2, c='b',  label='$1-e^{-1/T}$')
    #ax.plot(periodes, 1-threshold05, c='c', label='1-exp(-ts/T)')

    ax.legend(frameon=False)
    ax.set_ylabel('$k_{\epsilon}$')
    ax.set_xlabel('$T$')


def plot_threshold_functions_halfLife(var, par):
    fig, ax = plt.subplots()

    threshold1 = 2**(-1/(var.halfLifes))
    threshold2 = 1-2**(-1/(var.halfLifes*par.time_step))
    threshold05 = 2**(-par.time_step/(var.halfLifes))

    # ax.plot(periodes, threshold1, c='y', label ='exp(-1/T)')
    # ax.plot(periodes, threshold2, c='g',  label='exp(-1/(T*ts))')
    # ax.plot(periodes, threshold05, c='r', label='exp(-ts/T)')

    # ax.plot(periodes, 1-threshold1, c='m', label='1-exp(-1/T)')
    ax.plot(var.halfLifes, threshold2, c='b',  label='$1-2^{-1/t_{1/2}}$')
    # ax.plot(periodes, 1-threshold05, c='c', label='1-exp(-ts/T)')

    ax.legend(frameon=False)
    ax.set_ylabel('$k_{\epsilon}$')
    ax.set_xlabel('$t_{1/2}$')



def plot_threshold_functions_periode_halfLife():
    fig, ax = plt.subplots()

    threshold12 = np.exp(-halfLifes/(periodes*time_step))
    threshold2 = np.exp(-2*halfLifes/(periodes*time_step))
    threshold05 = np.exp(-halfLifes/(2*periodes*time_step))
    threshold22 = 2**(-2*halfLifes/(periodes*time_step))
    threshold14 = np.exp(-np.log(2)*halfLifes/(periodes*time_step))
    threshold214 = np.exp(-2*np.log(2)*halfLifes/(periodes*time_step))

    ax.plot(periodes, threshold12, c='b')
    ax.plot(periodes, threshold2, c='g')
    ax.plot(periodes, threshold05, c='r')
    ax.plot(periodes, threshold22, c='m')
    ax.plot(periodes, threshold14, c='y')
    #ax.plot(periodes, threshold214, c='c')

var = modelVar(periode, false_ratio, halfLife, kError, k0)
par = modelPar(time_step, vector_length, Nrealizations)
#plot_Period_halfLife_dep(var, par)

#stocastic_dependence =  create_stocastic_dependence(var, par)
noisy_dependence = create_stocastic_dependence(
    var, par)  # create_noisy_period_series(var, par)
plot_stocastic_dependence(noisy_dependence, var, par)

noisy_dependence =  create_noisy_period_series(var, par)
plot_stocastic_dependence(noisy_dependence, var, par)
values = rle(noisy_dependence)
print('valval', values, sum(values[0])) 
#trait_series, time_series = trait_evol(stocastic_dependence, var, par)
#print(trait_series)

#
#len_data_series = explore_periode_range(var, par)
#len_data_series = explore_noise_range(var, par)
#len_data_series = explore_halfLife_range(var, par)


#multiplot_NxM(m, n, par, var, hist_bins)

halfLife_values = [12, 22, 44, 88, 176, 356]

varNameY = 'halfLife'#'noiseLevel'#'kError'  # 'noiseLevel'#  # 'false_ratio'#
valuesY = var.halfLifes #var.noiseLevels  # var.kErrors  #  # var.falses_ratios#
varNameX = 'periode'  # 'noiseLevel'  # 'periode'
valuesX =  var.periodes#


#explore_twoVar_ranges(var, par, varNameX, valuesX, varNameY, valuesY)
plot_survival_martrix(varNameY, valuesY, varNameX, valuesX)

#multiplot_survivals(varNameY, valuesY, varNameX, valuesX, 'halfLife', halfLife_values)

#plot_threshold_functions_preiode(var, par)
#plot_threshold_functions_halfLife(var, par)
#max_death_interval(var, par)
#plot_threshold_functions_preiode()
#k_series_set, Dt_series_set, len_series_set = multiple_realizations(
#    Nrealizations, time_step, k0,  kError, periode, false_ratio,  halfLife)
#plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set)

# ethnobotanical low
Delta_t = 9 # [yr]
observed_per_surb = 100-9 # 100 - 8
name = 'ethnobotanical low'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(' percentage lost in time', Delta_t,
      '[yr] given half life = ', halfLife*par.time_step, loss_percent)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun)

# ethnobotanical high
Delta_t = 9  # [yr]
observed_per_surb = 100-26  # 100 - 8
name = 'ethnobotanical high'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(' percentage lost in time', Delta_t,
      '[yr] given half life = ', halfLife*par.time_step, loss_percent)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun)


#military test
Delta_t = 0.115#[yr]
observed_per_surb = 100-17#100 - 8
name = 'military'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(' percentage lost in time', Delta_t,
      '[yr] given half life = ', halfLife*par.time_step, loss_percent)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun)

# military perceptual
Delta_t = 1  # [yr]
observed_per_surb = 100-100*(80-52)/80  # 100 - 8
name = 'military perceptual'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun, '[yr]')

# military procedual-motor
Delta_t = 1  # [yr]
observed_per_surb = 100-100*(750-700)/750  # 100 - 8
name = 'military procedual-motor'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun, '[yr]')

# CPR
Delta_t = 3  # [yr]
observed_per_surb = 12  # 100 - 8
name = 'CPR'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun, '[yr]')

#gaelic football
Delta_t = 6*7/365 # [yr]
observed_per_surb = 100 - 100*(15-12)/15  # 100 - 8
name = 'gaelic'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun)

# recall school
Delta_t = 26*7/365  # [yr]
observed_per_surb = 100 - 100*(80-58)/80  # 100 - 8
name = 'recall school'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun)

# recognition school
Delta_t = 26*7/365  # [yr]
observed_per_surb = 100 - 100*(85-80)/85  # 100 - 8
name = 'gaelic'
loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
print(name, ' half life given percent of  surb  after', Delta_t,
      '[yr]= ', observed_per_surb, '%', halfLife_fun)



plt.show()








