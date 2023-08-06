import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.stats
from scipy.stats import binom
from numba import jit, float64, types, int64, bool_


@jit(types.Tuple((float64[:], float64[:]))(float64, float64, float64, int64), nopython=True, fastmath=True)
def build_train_pulses(ts, E0, tau, L2):

    time_array = np.linspace(0, ts, L2)
    E_values = E0 * np.exp(-time_array/(tau))
    if E_values[-1] < 1:
        E_values[-1] = 0
    else:
        E_values[-1] = E0

    return E_values, time_array


@jit(types.Tuple((int64, int64))(int64, float64, float64, float64, int64, int64, float64, float64), nopython=True, fastmath=True)
def countFracasos(nTimes, T, mu, sigma, L, L2, E0, tau):
    fracasos = 0

    for i in range(0, nTimes):
        ts_dist = T + np.random.normal(mu, sigma, L)
        for ts in ts_dist:

            E_values, t_values = build_train_pulses(ts, E0, tau, L2)

            if E_values[-1] == 0:
                fracasos += 1
                break

    fracasos2 = fracasos*2
    return fracasos, fracasos2


@jit((int64)(int64, float64, float64, float64, int64, float64), nopython=True, fastmath=True)
def countTempFracasos(nTimes, T, mu, sigma, Nper, t_death):
    fracasos = 0

    for i in range(0, nTimes):
        ts_dist = T + np.random.normal(mu, sigma, Nper)
        for ts in ts_dist:
            if ts >= t_death:
                fracasos += 1
                break
    return fracasos


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
   
    fig, ax = plt.subplots()
    ax.vlines(x=t_death, ymin=0, ymax=10, ls='-', linewidth=1.2, color='r')
    ax.hist(max_dist.flatten(), bins=33)
    #plt.hist(distances, bins=11)
    plt.show()
    return fracasos


def main():

    mu = 0
    T = 6
    L = 2000
    L2 = 100
    sigma = 2 * T
    Nper = int(L/T)

    tau = 7

    E0 = 44
    t_death = np.log(E0) * tau

    nTimes = 100000

    #fracasos, fracasos2 = countFracasos(nTimes, T, mu, sigma, Nper, L2, E0, tau)
    fracasosTemp = countTempFracasos(nTimes, T, mu, sigma, Nper, t_death)
    #print('sim Frac', fracasos/nTimes, 'Surb', 1-fracasos/nTimes)
    print('simTim Frac', fracasosTemp/nTimes, 'Surb', 1-fracasosTemp/nTimes)

    fracasosMeu = create_noisy_period_series(nTimes, L, T, sigma, t_death)
    print('simMeu Frac', fracasosMeu/nTimes, 'Surb', 1-fracasosMeu/nTimes)

    p_surb = scipy.stats.norm(T, sigma).cdf(t_death)
    p_surb_po = scipy.stats.poisson.cdf(T, sigma)
    p_death = 1- p_surb

    

    print('ts* = ', t_death, 'prob surb', p_surb)
    #binomial_pmf = 1-binom.pmf(0, Nper, p_death)
    cumulat_p_surb = p_surb**Nper
    #cumulat_p_death = (1-p_death)**Nper
    cumulat_p_surb_po = p_surb_po**L
    #print(binomial_pmf)
    print('analy norm Frac', 1-cumulat_p_surb, 'surb', cumulat_p_surb)
    #print('analy binom Frac', 1-cumulat_p_death,'Surb', cumulat_p_death)
    print('analy norm', 1-cumulat_p_surb_po, cumulat_p_surb_po)

    '''
    print('ts* = ', ts_death, 'prob death', p_death)
    print(fracasos)
    binomial_pmf = binom.pmf(1, L, p_death)
    print(binomial_pmf)
    '''
    # plt.plot(t_trajectory[1:], E_trajectory)
    # plt.axhline(y=1, c = 'r', ls = '--')
    # plt.show()


if __name__ == '__main__':
    main()
