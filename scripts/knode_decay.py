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
import plot_survival_matrices as psm
import plot_decay_functions as pdf
import name_files as nf
import Analitical_node as an
import scipy.stats


class modelVar:
    def __init__(self, periode, false_ratio, tau, kError, k0, tDeath):
        self.periode = periode
        self.periodes= periodes

        self.false_ratio = false_ratio
        self.falses_ratios = falses_ratios

        self.noiseLevel = noiseLevel
        self.noiseLevels = noiseLevels

        self.pop = pop
        self.pops = pops

        self.tau = tau
        self.taus = taus

        self.kError = kError
        self.kErrors = kErrors

        self.k0 = k0

        self.tDeath = tDeath
        self.tDeaths = tDeaths

class modelPar:
    def __init__(self, time_step, vector_length, Nrealizations):
        self.time_step = time_step
        self.vector_length = vector_length
        self.Nrealizations = Nrealizations
        self.output_dir = output_dir
        self.dropbox_dir = dropbox_dir
        self.root = root
        self.plots_dir = plots_dir
        self.plots_survivalMat = plots_survivalMat


def which_threshlod(var, par):

    if 'th-2t12' in par.root:
        return k0 * 2**(-2*var.tau/(var.periode ))
    elif 'th_pop' in par.root:
        return k0 * int(1/var.pop)
    elif 'th_pop1' in par.root:
        return k0 * 1
    elif 'th-t12' in par.root:
        return k0 * 2**(-var.tau/(var.periode))
    elif 'th-tau' in par.root:
        return k0 * np.exp(-var.tau*np.log(2)/(var.periode ))
    elif 'th-2tau' in par.root:
        return k0 * np.exp(-2*var.tau*np.log(2)/(var.periode ))
    elif 'th-e(-Tts)-1' in par.root:
        return k0 * np.exp(-1/(var.periode*par.time_step ))
    elif 'th-e(-t12ts)-1' in par.root:
        return k0 * np.exp(-1/(var.tau*par.time_step))
    elif 'th_1-e(-Tts)-1' in par.root:
        return k0 * (1- np.exp(-1/(var.periode*par.time_step)))
    elif 'th_1-e(-ts)-1' in par.root:
        return k0 * (1 - np.exp(-1/(var.tau*0.5)))
    elif 'th_1-e(-t12T)' in par.root:
        return k0 * ( 1- np.exp(-var.tau/(var.periode)))
    elif 'th_e(-t12T)' in par.root:
        return k0 * (np.exp(-var.tau/(var.periode)))
    elif 'th_e(-t12T)' in par.root:
        return k0 * (1-np.exp(-var.tau/(var.periode)))
    elif 'th_2(-Tt12)' in par.root:
        return k0 * (2**(-var.periode/(var.tau)))
    elif 'th_1-2(-Tt12)' in par.root:
        return k0 * (1-2**(-var.periode/(var.tau)))
    elif 'th_1-2(-t12ts)' in par.root:
        return k0 * (1-2**(-1/(var.tau*par.time_step)))
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
    
    kt = [var.pop*var.k0]
    kt_norm = [var.k0]
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
            #kt.append(kt[i-1]*np.exp(-t*var.tau))
            #kt.append(k0*np.exp(-t/(par.time_step*var.tau*np.log(2))))
            k = k0*np.exp(-t/(par.time_step*var.tau))#*np.log(2)
            discrete_k = int(var.pop * k)#/var.pop
            kt.append(discrete_k)
            kt_norm.append(discrete_k/var.pop)
            #kt.append(k0*2**(-t/(par.time_step*var.tau)))
        
            time_vec.append(t)
            #if i > 998:
            #    print('tttt it ended!', t, kt[i])
            
        if kt[i] > threshold: 
            kt.append(var.k0*var.pop)
            kt_norm.append(var.k0)
            time_vec.append(par.time_step)
        else: 
            #print("end of loop at ", i, kt[i], time_vec[i])
            break

        i = i+1
    
    return kt_norm , time_vec


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

    return stocastic_dependence


def create_noisy_period_series(var, par):

    # Create boolean vector
    stocastic_dependence = np.zeros(par.vector_length)  # , dtype=bool

    indexes = np.arange(0, par.vector_length, var.periode)
    #noise = np.random.random(len(indexes))*var.noiseLevel
    #print('is this raaaaight/?', var.noiseLevel)
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


def count_peridod_failure(var, par):

    # Create boolean vector
    stocastic_dependence = np.zeros(par.vector_length)  # , dtype=bool

    indexes = np.arange(0, par.vector_length, var.periode)
    #noise = np.random.random(len(indexes))*var.noiseLevel
    #print('is this raaaaight/?', var.noiseLevel)
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





def explore_oneVar_range(varName, varRange, var, par):

    len_data_series = []

    for v in varRange:
        name = nf.file_name_n_varValue(varName, v, var, par)
        print('name var', varName, 'val', v )
        k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
            multiple_noiseRealizations(var, par)
        len_data_series.append(np.array(len_series_set).flatten())
        #plot_stocastic_dependence(one_stocastic_dependence, var, par)
        #plot_traitTime_evol(one_trait_series, one_time_series)
        #plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, hist_bins)
        np.save(name, np.array(len_series_set).flatten())

    return len_data_series


def explore_twoVar_ranges(varNameX, varRangeX, varNameY, varRangeY, var, par ):

    for v in varRangeY:
        name = nf.file_name_n_varValue(varNameY, v, var, par)
        if os.path.exists(name):
            break
        #print('name var', varNameY, 'val', v)
        explore_oneVar_range(varNameX, varRangeX, var, par)


def analy_explore_twoVar_ranges(varNameX, varRangeX, varNameY, varRangeY, var, par, analy = ''):

    rows = len(varRangeY)
    cols = len(varRangeX)
    survival_mat = np.empty((rows, cols))
    survival_mat_alber = np.empty((rows, cols))
    to_modify = varNameX +'_'+ varNameY
    name_mat = nf.name_survival_ranges( to_modify, 'mat_', var, par, analy) + '.npy'
    name_mat_al = nf.name_survival_ranges( to_modify, 'mat_', var, par, 'alber') + '.npy'
    
    for i, v1 in enumerate(varRangeY):
        #if os.path.exists(name_mat):
        #    break
        n = nf.file_name_n_varValue(varNameY, v1, var, par, analy)
        
        for j, v2 in enumerate(varRangeX):
            n = nf.file_name_n_varValue(varNameX, v2, var, par, analy)
            #par.periode = v2
            ts_death = np.log(var.pop)*var.tau
            p_surb = scipy.stats.norm(var.periode, var.noiseLevel*var.periode).cdf(ts_death)
            
            cumulat_p_surb = p_surb**(par.vector_length/var.periode)
            n_frac_alber = an.countTempPositiveFracasos(
                par.Nrealizations, var.periode, 0, var.noiseLevel*var.periode, par.vector_length/var.periode, ts_death, par.vector_length)
            #print(var.periode, var.noiseLevel, 'dedede', cumulat_p_surb)
            survival_mat[i, j] = cumulat_p_surb
            survival_mat_alber[i, j] = 1-n_frac_alber/par.Nrealizations
    survival_mat[0, 0] = 1
    survival_mat_alber[0, 0] = 1
    #threshold_value = 0.95
    #count_higher = np.sum(survival_mat > threshold_value)
    #print( f"Number of elements higher than {threshold_value}: {count_higher} out of {len(survival_mat.flatten())}")
    np.save(name_mat, survival_mat)
    np.save(name_mat_al, survival_mat_alber)
    return name_mat, name_mat_al
        

def analy_explore_T_sigma_forTdeath(varNameX, varRangeX, varNameY, varRangeY, var, par, ts_death):

    rows = len(varRangeY)
    cols = len(varRangeX)
    survival_mat_analy = np.empty((rows, cols))
    survival_mat_sim = np.empty((rows, cols))
    to_modify = varNameX +'_'+ varNameY + '_' + 'tDeath'
    name_mat_analy = nf.name_survival_ranges( to_modify, 'mat_', var, par,  'analy_tth') + '.npy'
    name_mat_sim = nf.name_survival_ranges( to_modify, 'mat_', var, par, 'sim_tth') + '.npy'
    print('\n tDeath', ts_death, '\n')
    for i, v1 in enumerate(varRangeY):
        if os.path.exists(name_mat_sim):
            break
        n = nf.file_name_n_varValue(varNameY, v1, var, par, 'analy')
        
        for j, v2 in enumerate(varRangeX):
            n = nf.file_name_n_varValue(varNameX, v2, var, par, 'analy')
            
            p_surb = scipy.stats.norm(var.periode, var.noiseLevel*var.periode).cdf(ts_death)
            
            cumulat_p_surb = p_surb**(par.vector_length/var.periode)
            n_frac_sim = an.countTempPositiveFracasos(
                par.Nrealizations, var.periode, 0, var.noiseLevel*var.periode, par.vector_length/var.periode, ts_death, par.vector_length)
            #print(var.periode, var.noiseLevel, 'dedede', cumulat_p_surb)
            survival_mat_analy[i, j] = cumulat_p_surb
            survival_mat_sim[i, j] = 1-n_frac_sim/par.Nrealizations
    survival_mat_analy[0, 0] = 1
    survival_mat_sim[0, 0] = 1
    
    print( 'nanananame_mat_analy', name_mat_analy)
    np.save(name_mat_analy , survival_mat_analy )
    np.save(name_mat_sim, survival_mat_sim)
    return name_mat_analy, name_mat_sim
        
def periodMiss_twoVar_ranges(varNameX, varRangeX, varNameY, varRangeY, var, par):

    rows = len(varRangeY)
    cols = len(varRangeX)
    survival_mat = np.empty((rows, cols))
    to_modify = varNameX +'_'+ varNameY
    name_mat = nf.name_survival_ranges( to_modify, 'mat_', var, par, 'periodMiss') + '.npy'
    
    ts_death = np.log(var.pop)*var.tau

    print ('\n ts_death', ts_death, '\n')
    for i, v1 in enumerate(varRangeY):
        #if os.path.exists(name_mat):
        #    break
        n = nf.file_name_n_varValue(varNameY, v1, var, par)
        
        
        for j, v2 in enumerate(varRangeX):
            n = nf.file_name_n_varValue(varNameX, v2, var, par)
            #par.periode = v2
            

            #print('nper, ', int(par.vector_length/var.periode))
            
            failures = an.countPeriodFailures(par.Nrealizations, par.vector_length, 
                                              var.periode, var.false_ratio, ts_death)
            
            survival_mat[i, j] = 1-failures/par.Nrealizations
            
    np.save(name_mat, survival_mat)
    print('mamammiss', survival_mat)
    return name_mat
        



    
def explore_periode_range(var, par):

    len_data_series = []

    for T in var.periodes[2:4]:
        var.periode = int(T)
        var.kError = which_threshlod(var, par)
        print('pepeprpepreprpe', var.periode, 'nonono', var.noiseLevel )
        k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
            multiple_noiseRealizations(var, par)

        len_data_series.append(np.array(len_series_set).flatten())

        pdf.plot_stocastic_dependence(one_stocastic_dependence, var, par)
        pdf.plot_traitTime_evol(one_trait_series, one_time_series)
        pdf.plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, par, hist_bins)
        
        name = nf.file_name_n_varValue('periode',T)  # '  # str(var.kError) +
        np.save(name, np.array(len_series_set).flatten())
    
    return len_data_series

def explore_noise_range(var, par):

    len_data_series = []

    for n in var.falses_ratios:
        print('nnnnn', n)
        var.false_ratio = n

        explore_periode_range(var, par)

    return len_data_series

def explore_tau_range(var, par):


    for n in var.taus:
        print('ddddddd', n)
        var.tau = n

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


def multiexplore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, varNameZ, valuesZ, var, par):
    for e in valuesZ:
        name = nf.file_name_n_varValue(varNameZ, e, var, par)
        explore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, var, par)


def analy_multiexplore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, varNameZ, valuesZ, var, par, analy):
    for e in valuesZ:
        name = nf.file_name_n_varValue(varNameZ, e, var, par)
        analy_explore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, var, par, analy)

def analy_multiexplore_tDeath_range(varNameX, valuesX, varNameY, valuesY, tDeathName, valuestDeath, var, par, analy):
    for e in valuestDeath:
        name = nf.file_name_n_varValue(tDeathName, e, var, par)
        analy_explore_T_sigma_forTdeath(varNameX, valuesX, varNameY, valuesY, var, par, e)

def a_survival_martrix(varNameY, valuesY, varNameX, valuesX, var, par):

    survival_rate = np.empty([len(valuesY), len(valuesX)])
    for i, valY in enumerate(valuesY):
        nameY = nf.file_name_n_varValue(varNameY, valY, var, par)
        for j, valX in enumerate(valuesX):

            nameX = nf.file_name_n_varValue(varNameX, valX, var, par)
            dataset_len_series = np.load(nameX)
            survivors = len(np.where(dataset_len_series >
                            par.vector_length-10)[0])
            survival_rate[i][j] = survivors/par.Nrealizations

    survival_rate[-1][-1] = 0
    return survival_rate
    # plt.savefig(name_fig, bbox_inches='tight')


# python scripts/knode_decay.py 'th_1-e(-Tts)-1' 44
'''
what would be, eventually, a trait tau inside a knode, this code shall be eficient, fast, repeated

called "drifwood model" as it tries to model what is the threshold that the presence of a resouce/dependence
has to occur in order for the trait to remain "knowlwgelable" onside of the knode

'''
# visualization par
hist_bins = 18

# numerical par
time_step = 1
Nrealizations = 333
vector_length = int(330/time_step)

# model par
output_dir = './data/output/'
root = sys.argv[1]  # 'th_pop' # 'noisTpc_th-e-(Tts)-1'  # 'noisTpc_th-2t12'
plots_dir = './plots/embers/'
plots_survivalMat = 'fig_survivalMatrix/'
dropbox_dir = '/Users/au710647/Desktop/Dropbox/cultural_loss_project/embers'

periode = 4/time_step  # int(-time_step*np.log(kError)/tau)
max_period = int(25/time_step)  # int(-time_step*np.log(kError)/tau)
min_period = 2  # do never go below 1 for ressolution issues.
step_period = (max_period - min_period)/50.
periodes = np.arange(min_period, max_period, step_period)
#print('peripperi', periodes)

tau = float(sys.argv[2])  # 88#np.log(2)/periode  # 0.05  #
max_tau = (20/time_step)  # int(-time_step*np.log(kError)/tau)
min_tau = 2   # do never go below 1 for ressolution issues.
step_tau = (max_tau - min_tau)/5.
taus =  np.arange(min_tau, max_tau, step_tau)
#print('dededededekay', taus)


k0 = 1  # initial knowledge
kError = k0 * np.exp(-tau/(periode*time_step))  # 1/10.*k0
min_kError = 0.01
max_kError = 0.95
step_kError = (max_kError - min_kError)/11
kErrors = np.append(np.array([min_kError]), np.arange(
    min_kError, max_kError, step_kError))
#print('kekekekkee', kErrors)

false_ratio = 0.11
max_falses = 0.9  # 0.6#0.66
min_falses = 0.01
step_falses = (max_falses - min_falses)/50.
falses_ratios = np.arange(min_falses, max_falses, step_falses)
#print('falsfalsfalse', falses_ratios)

noiseLevel = 0.3
max_noises = 2.1  # 0.0002  # 0.6#0.66
min_noises = 0.1  # 0.0001
step_noises = (max_noises - min_noises)/50.
noiseLevels = np.arange(min_noises, max_noises, step_noises)
##print('nisnosisnoise', noiseLevels)

pop = int(sys.argv[3])  # 33
max_pop = 51  # 0.0002  # 0.6#0.66
min_pop = 1  # 0.0001
step_pop = (max_pop - min_pop)/11.
pops = np.arange(min_pop, max_pop, step_pop)
#print('popopopopopo', pops)

tDeath = 15.1
max_tDeath = 64
min_tDeath = 2
step_tDeaths = (max_pop - min_pop)/11.
tDeaths = np.arange(min_pop, max_pop, step_pop)



def main():
    ##python knode_decay.py th_pop1 11 44
    var = modelVar(periode, false_ratio, tau, kError, k0, tDeath)
    par = modelPar(time_step, vector_length, Nrealizations)
    #plot_Period_tau_dep(var, par)


    ''' noisy dependence erases 1 every n events in a perfect period '''
    noisy_dependence = create_stocastic_dependence( var, par)  
    pdf.plot_stocastic_dependence(noisy_dependence, var, par)
    #noisy_dependence =  create_noisy_period_series(var, par)
    
    k_series_set, Dt_series_set, len_series_set, one_stocastic_dependence, one_trait_series, one_time_series =\
        multiple_noiseRealizations(var, par)
    
    #fracasos, E_series, events = an.countFracasos(par.Nrealizations, var.periode, 0, var.noiseLevel*var.periode, int(par.vector_length/var.periode), 100, var.pop, var.tau)
     
    pdf.plot_stocastic_dependence(one_stocastic_dependence, var, par)
    pdf.plot_traitTime_evol(one_trait_series, one_time_series, var, par)
    pdf.plot_traitTime_evol_and_noise_sequence(
        one_trait_series, one_stocastic_dependence, var, par)
    #pdf.plot_traitTime_evol_and_noise_sequence(
    #    E_series, events, var, par)
    values = rle(noisy_dependence)
    #print('valval', values, sum(values[0])) 
    
    #trait_series, time_series = trait_evol(stocastic_dependence, var, par)
    #print(trait_series)

    #len_data_series = explore_periode_range(var, par)
    #len_data_series = explore_noise_range(var, par)
    #len_data_series = explore_tau_range(var, par)

    #tau_values = [8, 10, 12, 16, 20, 24]
    #tau_values = [4, 8, 12, 16, 20]
    tau_values = [1, 2, 4, 8, 16]
    pop_values = [ 3, 6, 12, 24, 48]
    tDeath_values = [2, 4, 8, 16, 32, 64]

    varNameY = 'noiseLevel'  # 'noiseLevel'#'kError'  # 'noiseLevel'#  # 'false_ratio'#
    valuesY = var.noiseLevels #var.noiseLevels  # var.kErrors  #  # var.falses_ratios#
    varNameX = 'periode'  # 'noiseLevel'  # 'periode'
    valuesX =  var.periodes#
    varNameZ = 'tau'  # 'noiseLevel'  # 'periode'
    valuesZ = tau_values #var.taus
    varNameS = 'pop'  
    valuesS = pop_values
    varNameM = 'false_ratio'  
    valuesM = var.falses_ratios
    tDeathName = 'tDeath'
    valuestDeath = var.tDeaths

    ''' Here to plot a matrix with failure simulation '''
    #explore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, var, par)
    #psm.plot_survival_martrix(varNameY, valuesY, varNameX, valuesX, var, par)

    ''' Here to plot a matrix of period misses'''
    #name_mat_perMiss = periodMiss_twoVar_ranges(varNameX, valuesX, varNameM, valuesM, var, par)
    #psm.plot_survival_martrix(varNameM, valuesM, varNameX, valuesX, var, par)
    #psm.plot_analy_survival_matrix(
    #  varNameM, valuesM, varNameX, valuesX, name_mat_perMiss, var, par, '')
    
    ''' Here to plot a matrix with simulatiocountPeriodFailuresn and analyitical'''
    #analy = ''
    #name_mat_analy, name_mat_albert = analy_explore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, var, par, analy)
    #psm.plot_analy_survival_matrix(
    #    varNameY, valuesY, varNameX, valuesX, name_mat_albert, var, par, analy)
    #analy = 'analy'
    #psm.plot_analy_survival_matrix(
    #    varNameY, valuesY, varNameX, valuesX, name_mat_analy, var, par, analy)
    #psm.plot_simAndAnaly_survival_matrix(
    #    varNameY, valuesY, varNameX, valuesX, name_mat_albert, name_mat_analy,  var, par, analy)
    
    ''' Here to plot the grid of matrices with my method'''
    #multiexplore_twoVar_ranges(varNameX, valuesX,
    #                          varNameY, valuesY, varNameZ, valuesZ, var, par)

    ''' Here to plot the grid of matrices with tDeath dependence'''
    analy = 'analy'
    analy_multiexplore_tDeath_range(varNameX, valuesX, varNameY, valuesY, tDeathName, valuestDeath, var, par, analy)
    psm.colum_multiplot_analy_survivals(varNameY, valuesY, varNameX, valuesX, tDeathName, valuestDeath, 'analy_tth', var, par)
    plt.show()
    
    ''' Here to plot the grid of matrices with simulation and analyitical'''
    
    #for v in valuesS:
    #    name = nf.file_name_n_varValue(varNameS, v, var, par)
    #    analy_multiexplore_twoVar_ranges(varNameX, valuesX, varNameY, valuesY, varNameZ, valuesZ, var, par, analy)
    #psm.colum_multiplot_analy_survivals(
    #    varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, analy, var, par)
    #psm.multiplot_mxn_analy_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, varNameS, valuesS, var, par)
    #psm.multiplot_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, var, par)



    #plot_threshold_functions_preiode(var, par)
    #plot_threshold_functions_tau(var, par)
    #max_death_interval(var, par)
    #plot_threshold_functions_preiode()
    #k_series_set, Dt_series_set, len_series_set = multiple_realizations(
    #    Nrealizations, time_step, k0,  kError, periode, false_ratio,  tau)
    #plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set)





if __name__ == '__main__':
    main()
    #plt.show()






