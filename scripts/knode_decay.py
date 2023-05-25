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
vector_length = 2000


#model var
output_dir = './data/output/'
root = 'noiseLevel_threshold207/'
plots_dir = './plots/embers/'



periode = 44/time_step#int(-time_step*np.log(kError)/decay)
max_period = int(83/time_step)#int(-time_step*np.log(kError)/decay)
min_period = 2 # do never go below 1 for ressolution issues. 
step_period = (max_period - min_period)/11.
periodes= np.arange(min_period, max_period, step_period)
print('peripperi', periodes)

decay = 0.016#np.log(2)/periode  # 0.05  # np.log(kError)/peride
max_decay = 0.050  # int(-time_step*np.log(kError)/decay)
min_decay = 0.005   # do never go below 1 for ressolution issues.
step_decay = (max_decay - min_decay)/111
decays = np.arange(min_decay, max_decay, step_decay)
print('dededededekay', decays)


k0 = 1  # initial knowledge
kError = k0 * np.exp(periode*decay*time_step)  # 1/10.*k0
min_kError = 0.05
max_kError = 0.95
step_kError = (max_kError - min_kError)/11
kErrors = np.arange(min_kError, max_kError, step_kError)
print('kekekekkee', kErrors)

false_ratio = 0.11
max_falses = 0.02#0.6#0.66
min_falses = 0.01  
step_falses = (max_falses - min_falses)/1.
falses_ratios = np.arange(min_falses, max_falses, step_falses)
print('falsfalsfalse', falses_ratios)

noiseLevel = 0.5
max_noises = 5.1#0.0002  # 0.6#0.66
min_noises = 0.1#0.0001
step_noises = (max_noises - min_noises)/11.
noiseLevels = np.arange(min_noises, max_noises, step_noises)
print('nisnosisnoise', noiseLevels)
#visualization par
hist_bins = 18




class modelVar:
    def __init__(self, periode, false_ratio, decay, kError, k0):
        self.periode = periode
        self.periodes= periodes

        self.false_ratio = false_ratio
        self.falses_ratios = falses_ratios

        self.noiseLevel = noiseLevel
        self.noiseLevels = noiseLevels

        self.decay = decay
        self.decays = decays

        self.kError = kError
        self.kErrors = kErrors
        self.k0 = k0


class modelPar:
    def __init__(self, time_step, vector_length, Nrealizations):
        self.time_step = time_step
        self.vector_length = vector_length
        self.Nrealizations = Nrealizations


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
    threshold = k0 * np.exp(-2*np.log(2)/(var.periode*var.decay*time_step))#var.kError
    while i < len(stocastic_dependence)-1:
        
        t = time_step
        #if i > 988:
        #    print('iiiuuuu', i, t)
        while i < len(stocastic_dependence)-1 and stocastic_dependence[i] == 0 and kt[i] > threshold: 
            t = t + time_step
            i = i+1
            #kt.append(kt[i-1]*np.exp(-t*var.decay))
            kt.append(k0*np.exp(-t*var.decay))
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

    for T in var.periodes:
        var.periode = int(T)
        print('pepeprpepreprpe', var.periode, 'nonono', var.false_ratio )
        k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
            multiple_noiseRealizations(var, par)

        len_data_series.append(np.array(len_series_set).flatten())

        plot_stocastic_dependence(one_stocastic_dependence, var, par)
        plot_traitTime_evol(one_trait_series, one_time_series)
        plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, hist_bins)
        
        name = output_dir +'/' + root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(T) + '_Te=' +\
            "{:.2f}".format(var.false_ratio) + '_ke=' +  \
            '_d=' + "{:.2}".format(var.decay)+'.npy'  # str(var.kError) +
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

        #name = output_dir + root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(var.periode) + '_Te=' +\
        #    "{:.2f}".format(n) + '_ke=' + str(var.kError) + \
        #    '_d=' + "{:.2}".format(var.decay)+'.npy'
        #np.save(name, np.array(len_series_set).flatten())

    return len_data_series



def explore_decay_range(var, par):


    for n in var.decays:
        print('ddddddd', n)
        var.decay = n

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


def file_name_n_varValue(to_modify, value):

    if to_modify == 'periode':
        var.periode = value
        
    elif to_modify == 'decay':
        var.decay = value
        
    elif to_modify == 'false_ratio':
        var.false_ratio = value

    elif to_modify == 'noiseLevel':
        var.noiseLevel = value
        
    elif to_modify == 'kError':
        var.kError = value
        
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, decay, false_ratio, kError')
        quit()

    #name = output_dir + root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(var.periode) + '_Te=' +\
    #    "{:.2f}".format(var.false_ratio) + '_ke=' + str(var.kError) + \
    #   '_d=' + "{:.2}".format(var.decay)+'.npy'
    
    name = output_dir + root+'_ts='+str(par.time_step)+'T=' + "{:.2f}".format(var.periode) + '_Te=' +\
        "{:.2f}".format(var.noiseLevel) + '_ke='  + \
        '_d=' + "{:.2}".format(var.decay)+'.npy'  # + str(var.kError)
    
    return name


def name_survival_fig(to_modify, root_fig):

    
    if 'periode' in to_modify:
        periode_seg = '_Tran-'+"{:.2f}".format(var.periodes[0])+'-'+"{:.2f}".format(var.periodes[-1])
    else:
        periode_seg = '_T-' + "{:.2f}".format(var.periode)

    if 'decay' in periode_seg:
        decay_seg = '_Dran-'+"{:.2}".format(var.decays[0])+'-'+"{:.2}".format(var.decays[-1])
    else:
        decay_seg = '_D-' + "{:.2}".format(var.decay)

    if 'false_ratio' in to_modify:
        false_ratio_seg = '_FRran-'+"{:.2f}".format(var.falses_ratios[0])+' -'+"{:.2f}".format(var.falses_ratios[-1])
    else:
        false_ratio_seg = '_FR-' + "{:.2f}".format(var.false_ratio)

    if 'noiseLevel' in to_modify:
        noiseLevel_seg = '_TNran-'+"{:.2f}".format(var.noiseLevels[0])+' -'+"{:.2f}".format(var.noiseLevels[-1])
    else:
        noiseLevel_seg = '_TN-' + "{:.2f}".format(var.noiseLevel)

    if  'kError' in to_modify:
        kError_seg = '_kEran-'+"{:.2f}".format(var.kErrors[0])+' -'+"{:.2f}".format(var.kErrors[-1])
    else:
        kError_seg = '_kE-' + "{:.2f}".format(var.kError)


    name_fig = plots_dir + root_fig + root +'_ts='+str(par.time_step) + '_L='+str(par.vector_length)+ \
        periode_seg + decay_seg + noiseLevel_seg + kError_seg + '.png'

    return name_fig 


def var_tagAndLabels(varName, values):

    labels = []

    if varName == 'periode':
        tag = '$T$[yr]'
        for e in values:
            labels.append("{:.0f}".format(e*par.time_step))

    elif varName == 'decay':
        tag = '$t_{1/2}$[yr]'
        #tag = '$\lambda$[yr$^-1$]'
        for e in values:
            labels.append("{:.2}".format(np.log(2)*par.time_step/e))

    elif varName == 'false_ratio':
        tag='$T_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2f}".format(e))

    elif varName == 'noiseLevel':
        tag = '$T_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2f}".format(e))

    elif varName == 'kError':
        tag = '$k_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2f}".format(e))
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, decay, false_ratio, kError')
        quit()

    return tag, labels


def plot_survival_martrix(varNameY, valuesY, varNameX, valuesX, plots_survivalMat='fig_survivalMatrix/'):

    survival_rate = np.empty([len(valuesY), len(valuesX)])
    for i, valY in enumerate(valuesY):
        nameY = file_name_n_varValue(varNameY, valY)
        for j, valX  in enumerate(valuesX):
            print('noise', "{:.2f}".format(var.false_ratio), 'per', "{:.2f}".format(var.periode))

            nameX = file_name_n_varValue(varNameX, valX)
            dataset_len_series = np.load(nameX)
            # print('ufufufuf', dataset_len_series)
            survivors = len(np.where(dataset_len_series >
                            par.vector_length-10)[0])
            survival_rate[i][j] = survivors/par.Nrealizations
            print('sisisiusususu', survivors/par.Nrealizations)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(survival_rate, cmap='OrRd')  # extent = extent, 'bone'

    tagX, labelsX = var_tagAndLabels(varNameX, valuesX)
    tagY, labelsY = var_tagAndLabels(varNameY, valuesY)

    ax.set(xticks=np.arange(len(valuesX)), xticklabels=labelsX,
           yticks=np.arange(len(valuesY)), yticklabels=labelsY)
    
    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)

    fig.colorbar(im, cax=cax, orientation='vertical')   

    to_modify = varNameX + '_' + varNameY
    name_fig = name_survival_fig(to_modify, plots_survivalMat)

    #ax.plot(par.vector_length *
    #        0.005/(var.periodes*par.time_step)**2)

    print('nananannana', name_fig)
    plt.savefig(name_fig, bbox_inches='tight')
 

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


def max_death_interval(var, par):

    n_max_mat = []
    for e in var.kErrors:
        n_max_row = []
        for l in var.decays:
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
    x_tic_labels = np.log(2)*par.time_step/var.decays
    im = ax.contourf(x_tic_labels, var.kErrors, n_max_mat,
                     extend="both", cmap='bone')

    tagX, labelsX = var_tagAndLabels('decay', var.decays)
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

    #ax.set(xticks=np.arange(0, len(var.decays), intervalX), xticklabels=labelsX_short,
    #       yticks=np.arange(0, len(var.kErrors), intervalY), yticklabels=labelsY_short)

    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)


    fig.colorbar(im, cax=cax, orientation='vertical')



    return n_max_mat


def max_halfLife_interval(var, par):

    halfLife_mat = []
    for e in var.kErrors:
        halfLife_row = []
        for l in var.periodes:
            halfLife_row.append(par.time_step * np.log(k0/e)/l)
        halfLife_mat.append(halfLife_row)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    n_max_mat = np.array(halfLife_mat)
    aux = np.where(halfLife_mat > 86)
    n_max_mat[aux] = 0
    aux = np.where(halfLife_mat < 2)
    n_max_mat[aux] = 86

    # im = ax.imshow(n_max_mat, cmap='bone')  # extent = extent, 'bone'
    x_tic_labels = np.log(2)*par.time_step/var.decays
    im = ax.contourf(x_tic_labels, var.kErrors, halfLife_mat,
                     extend="both", cmap='bone')

    tagX, labelsX = var_tagAndLabels('periodes', var.periodes)
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

    # ax.set(xticks=np.arange(0, len(var.decays), intervalX), xticklabels=labelsX_short,
    #       yticks=np.arange(0, len(var.kErrors), intervalY), yticklabels=labelsY_short)

    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)

    fig.colorbar(im, cax=cax, orientation='vertical')

    return halfLife_mat
 
def plot_threshold_functions():
    threshold = np.exp(-1/(periodes*decay*time_step))
    threshold2 = np.exp(-2/(periodes*decay*time_step))
    threshold05 = np.exp(-1/(2*periodes*decay*time_step))
    threshold22 = 2**(-2/(periodes*decay*time_step))
    threshold14 = np.exp(-np.log(2)/(periodes*decay*time_step))
    threshold214 = np.exp(-2*np.log(2)/(periodes*decay*time_step))

    plt.plot(periodes, threshold, c = 'b')
    plt.plot(periodes, threshold2, c = 'g')
    plt.plot(periodes, threshold05, c = 'r')
    plt.plot(periodes, threshold22, c='m')
    plt.plot(periodes, threshold14, c='y')
    plt.plot(periodes, threshold214, c='c')

var = modelVar(periode, false_ratio, decay, kError, k0)
par = modelPar(time_step, vector_length, Nrealizations)
#plot_Period_decay_dep(var, par)

#stocastic_dependence =  create_stocastic_dependence(var, par)
noisy_dependence = create_noisy_period_series(var, par)
values = rle(noisy_dependence)
print('valval', values, sum(values[0])) 
#trait_series, time_series = trait_evol(stocastic_dependence, var, par)
#print(trait_series)


#len_data_series = explore_periode_range(var, par)
#len_data_series = explore_noise_range(var, par)
#len_data_series = explore_decay_range(var, par)


#multiplot_NxM(m, n, par, var, hist_bins)

varNameY = 'noiseLevel'#'kError'  # 'noiseLevel'#'decay'  # 'false_ratio'#
valuesY = var.noiseLevels#var.kErrors  # var.decays  # var.falses_ratios#
varNameX = 'periode'  # 'noiseLevel'  # 'periode'
valuesX =  var.periodes#

plot_threshold_functions()

explore_twoVar_ranges(var, par, varNameX, valuesX, varNameY, valuesY)
plot_survival_martrix(varNameY, valuesY, varNameX, valuesX)
#max_death_interval(var, par)
#k_series_set, Dt_series_set, len_series_set = multiple_realizations(
#    Nrealizations, time_step, k0,  kError, periode, false_ratio,  decay)
#plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set)

print('Half Life $t_{1/2}$= ', np.log(2)/var.decay*0.5 )

plt.show()








