import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.stats
from scipy.stats import binom
from numba import jit, float64, types, int64, bool_


@jit(types.Tuple((float64[:], float64[:]))(float64, float64, float64, int64), nopython=True, fastmath=True)
def build_train_pulses(ts, E0, tau, L2):

    #time_array = np.linspace(0, ts, L2)
    time_array = np.arange(0, ts, 1)
    E_values = E0 * np.exp(-time_array/(tau/2))

    if E_values[-1] < 1:
        E_values[-1] = 0
    #else:
    #    E_values[-1] = E0

    return E_values, time_array




#@jit(types.Tuple((int64, int64))(int64, float64, float64, float64, int64, int64, float64, float64), nopython=True, fastmath=True)
def countFracasos(nTimes, T, mu, sigma, nPer, L2, E0, tau):
    fracasos = 0

    E_values_serie = [E0]
    time_array_serie = [0]
    for i in range(0, nTimes):
        ts_dist =  np.random.normal(T, sigma, nPer)

        aux = np.where(ts_dist > 0)
        positive_ts_dist = ts_dist[aux]

        for ts in positive_ts_dist:
            
            E_values, t_values = build_train_pulses(ts, E0, tau, L2)

            if i == 1:
                E_values_serie = np.append(E_values_serie, E_values)
                time_array_serie = np.append(time_array_serie, t_values + time_array_serie[-1] ) 

            if E_values[-1] == 0:
                fracasos += 1
                break

    aux = np.where(E_values_serie == E0)
    events = np.zeros(len(E_values_serie))
    events[aux] = 1

    return fracasos, E_values_serie[1:], events[1:]


def generateDecaySeries(eventsTimes, L2, E0, tau):
    fracasos = 0

    E_values_serie = [E0]
    for t in  eventsTimes:
        E_values, t_values = build_train_pulses(t, E0, tau, L2)
        E_values_serie = np.append(E_values_serie, E_values) 

        if len(E_values_serie) > 2000:
            break
    
    return E_values_serie


@jit((int64)(int64, float64, float64, float64, int64, float64), nopython=True, fastmath=True)
def countTempFracasos(nTimes, T, mu, sigma, Nper, t_death):
    fracasos = 0

    for i in range(0, nTimes):
        ts_dist =  np.random.normal(T, sigma, Nper)
        for ts in ts_dist:
            if ts >= t_death:
                fracasos += 1
                break
            
    return fracasos

#    v_I = np.empty((enum, bnum), dtype=np.float64)

@jit((int64)(int64, float64, float64, float64, int64, float64, int64), nopython=True, fastmath=True)
def countTempPositiveFracasos(nTimes, T, mu, sigma, Nper, t_death, timeLim):
    fracasos = 0
    timeLength = 0
    for i in range(0, nTimes):
        ts_dist = np.random.normal(T, sigma, Nper)
        aux = np.where(ts_dist>0)
        ts_dist_pos = ts_dist[aux]
        for ts in ts_dist_pos:
            timeLength =+ ts
            if ts >= t_death:
                fracasos += 1
                break
            if timeLength > timeLim:
                break
            
    return fracasos


def countVarFracasos(nTimes, sigma, Nper, t_death):
    fracasos = 0

    for i in range(0, nTimes):
        ts_dist = np.random.normal(0, sigma, Nper)
        for ts in ts_dist:
            if ts >= t_death:
                fracasos += 1
                break
    return fracasos


#@jit(types.Tuple((int64, int64))(int64, float64, float64, int64, float64), nopython=True, fastmath=True)
def create_stocastic_dependence(nTimes, false_ratio, periode, vector_length, t_death ):

    # Create boolean vector
    stocastic_dependence = np.zeros(vector_length)  # , dtype=bool

    # Set false values at given intervals
    count_gaps = 0
    fracasos = 0
    for j in (0, nTimes):
        if j < 50:
                print('jjjj', j)
        for i in range(0, vector_length, int(periode)):
            #if i < 50:
            #    print('iiii', i, 'jjjj', j)
            if np.random.random() > false_ratio:
                count_gaps = 0
                #stocastic_dependence[i] = 1
            else: 
                count_gaps += 1
                if count_gaps*periode >= t_death:
                    fracasos += 1
                    break

    return fracasos#, stocastic_dependence




#@jit((int64)(int64, int64, float64, float64, float64), nopython=True, fastmath=True)
def create_noisy_period_series(nTimes, L, T, sigma, t_death):

    fracasos = 0
    max_dist = np.arange(nTimes)

    for n in range(nTimes):
        # Create boolean vector
        stocastic_dependence = np.zeros(L)  

        indexes = np.arange(0, L, T)
        #noise = np.random.normal(loc=0, scale=sigma, size=len(indexes))
        noise = np.random.normal(0, sigma, len(indexes))

        #sum = noise.astype('int') + indexes.astype('int')
        sum = noise + indexes
        index = np.where((sum >= 0) & (sum < L))
        #if n == 1:
            #print(sum[:11])
            #print(sum[index].astype('int'))
        

        stocastic_dependence[sum[index].astype('int')] = 1
        
        distances = np.diff(np.where(stocastic_dependence > 0))#-1
        any = np.where(distances[0] >= t_death)
        if len(any[0]) >= 1:
            fracasos = fracasos + 1
        
        max_dist[n] =  np.max(distances)

        

        #print('sssss', distances[0][:11], 'frac', fracasos, 'any?', len(any[0]), any[0])
   
    #fig, ax = plt.subplots()
    #ax.vlines(x=t_death, ymin=0, ymax=10, ls='-', linewidth=1.2, color='r')
    #ax.hist(max_dist.flatten(), bins=33)
    #plt.hist(distances, bins=11)
    #plt.show()
    return fracasos


def create_noisy_period_series_var(nTimes, L, sigma, t_death):

    fracasos = 0
    max_dist = np.arange(nTimes)

    for n in range(nTimes):
        # Create boolean vector
        stocastic_dependence = np.zeros(L)  

        indexes = np.arange(0, L, sigma)
        #noise = np.random.normal(loc=0, scale=sigma, size=len(indexes))
        noise = np.random.normal(0, sigma, len(indexes))

        #sum = noise.astype('int') + indexes.astype('int')
        sum = noise + indexes
        index = np.where((sum >= 0) & (sum < L))
        #if n == 1:
            #print(sum[:11])
            #print(sum[index].astype('int'))
        

        stocastic_dependence[sum[index].astype('int')] = 1
        
        distances = np.diff(np.where(stocastic_dependence > 0))#-1
        any = np.where(distances[0] >= t_death)
        if len(any[0]) >= 1:
            fracasos = fracasos + 1
        
        max_dist[n] =  np.max(distances)

        

        #print('sssss', distances[0][:11], 'frac', fracasos, 'any?', len(any[0]), any[0])
   
    #fig, ax = plt.subplots()
    #ax.vlines(x=t_death, ymin=0, ymax=10, ls='-', linewidth=1.2, color='r')
    #ax.hist(max_dist.flatten(), bins=33)
    #plt.hist(distances, bins=11)
    #plt.show()
    return fracasos

def plot_exp_dec(tau, E0):
   

    t_death = np.log(E0) * tau

    time = np.arange(int(t_death+1.5))

    exp = E0*np.exp(-time/tau)

    fig, ax = plt.subplots()
    plt.annotate('$\Delta t_{th}$', xy=(t_death/2, E0-3), fontsize = 13)
    plt.arrow(0, E0, t_death, 0,  linewidth=2,  head_width=1.05, head_length=1.03, color='k', length_includes_head=True)#t_death,
    plt.arrow(t_death, E0, -t_death, 0,  linewidth=2,  head_width=1.05, head_length=1.03, color='k', length_includes_head=True)#t_death,


    ax.vlines(x=t_death, ymin=0, ymax=E0, ls='-', linewidth=1.8, color='r')
    ax.plot(time, exp,  linewidth=1.8)
    #plt.hist(distances, bins=11)
    ax.set_ylabel('$N_e(t)$', fontsize = 13)
    ax.set_xlabel('t[yr]', fontsize = 13)


def main():

    mu = 0
    T = 4
    L = 1000
    L2 = 100
    sigma = 2 * T
    false_ratio = 0.5
   
    Nper = int(L/T)+L2

    tau = 4.5

    E0 = 44
    plot_exp_dec(T, E0)


    t_death = np.log(E0) * tau

    nTimes = 1000

    fracasos, E_series= countFracasos(nTimes, T, mu, sigma, Nper, L2, E0, tau)
    fracasosTemp = countTempFracasos(nTimes, T, mu, sigma, Nper, t_death)
    fracasosPosTemp = countTempPositiveFracasos(nTimes, T, mu, sigma, Nper, t_death, L)
    #E_series = generateDecaySeries(ts_dist_pos, L2, E0, tau)
    #print('sim Frac', fracasos/nTimes, 'Surb', 1-fracasos/nTimes)
    print('simTim Frac', fracasosTemp/nTimes, 'Surb', 1-fracasosTemp/nTimes)
    print('simPosTim Frac', fracasosPosTemp/nTimes, 'Surb', 1-fracasosPosTemp/nTimes)

    fracasosMeu = create_noisy_period_series(nTimes, L, T, sigma, t_death)
    print('simMeu Frac', fracasosMeu/nTimes, 'Surb', 1-fracasosMeu/nTimes)

    
    p_surb = scipy.stats.norm(T, sigma).cdf(t_death)
    p_surb_po = scipy.stats.poisson.cdf(T, sigma)
    p_death = 1- p_surb

    fig, ax = plt.subplots()
    print('t_series', len(E_series))
    #ax.plot(ts_dist_pos[1:], E_series[1:]) 
    ax.plot(E_series) 

    ax.plot(events)

    print()
    #plt.show()

    

    print('ts* = ', t_death, 'prob surb', p_surb)
    #binomial_pmf = 1-binom.pmf(0, Nper, p_death)
    cumulat_p_surb = p_surb**Nper
    #cumulat_p_death = (1-p_death)**Nper
    cumulat_p_surb_po = p_surb_po**L
    #print(binomial_pmf)
    print('analy norm Frac', 1-cumulat_p_surb, 'surb', cumulat_p_surb)
    #print('analy binom Frac', 1-cumulat_p_death,'Surb', cumulat_p_death)
    print('analy norm', 1-cumulat_p_surb_po, cumulat_p_surb_po)

    sigma2 = 6
    fracasosTempVar = create_noisy_period_series_var(nTimes, L, sigma2, t_death)
    Nper2 = int(L/sigma2)
    p_surb_var = scipy.stats.norm(0, sigma2).cdf(t_death)
    cumulat_p_surb_var = p_surb_var**Nper
    print('\nsim Var Frac', fracasosTempVar/nTimes, 'Surb', 1-fracasosTempVar/nTimes)
    print('analy Var Frac', 1-cumulat_p_surb_var, 'surb', cumulat_p_surb_var, '\n')

    fracasos_fail = create_stocastic_dependence(nTimes, false_ratio, T, L, t_death)
    death_gap = int(t_death/T)
    
    print('\nsim Fail Frac', fracasos_fail/nTimes, 'Surb', 1-fracasos_fail/nTimes)
    print(death_gap, nTimes, 'prob Fail, surbibal', false_ratio**death_gap*int(L/T), int(L/T)   )
    

    '''
    print('ts* = ', ts_death, 'prob death', p_death)
    print(fracasos)
    binomial_pmf = binom.pmf(1, L, p_death)
    print(binomial_pmf)
    '''
    #plt.plot(t_trajectory[1:], E_trajectory)
    # plt.axhline(y=1, c = 'r', ls = '--')
    plt.show()




if __name__ == '__main__':
    main()
