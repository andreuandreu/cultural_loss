import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.stats
from scipy.stats import binom
from numba import jit, float64, types, int64, bool_


@jit(types.Tuple((float64[:], float64[:]))(float64, float64, float64),nopython=True,fastmath=True)
def build_train_pulses(ts, E0, tau):

    L2 = 100
    time_array = np.linspace(0, ts, L2)
    E_values = E0 * np.exp(-time_array/tau)
    if E_values[-1] < 1:
        E_values[-1] = 0
    else:
        E_values[-1] =  E0

    return E_values, time_array 

@jit(types.Tuple((int64, int64))(int64, float64, float64, float64, int64, float64, float64),nopython=True,fastmath=True)
def countFracasos(nTimes, T, mu, sigma, L, E0, tau):
    fracasos = 0
    
    for i in range(0, nTimes):
        ts_dist = T + np.random.normal(mu, sigma, L)
        for ts in ts_dist:

            E_values, t_values = build_train_pulses(ts, E0, tau)
            if E_values[-1] == 0:
                fracasos += 1 
                break

    fracasos2 = fracasos*2
    return fracasos, fracasos2


def main():

    mu = 0; sigma = 1
    T = 6.2; L = 50
    
    tau = 2; threshold = 1
    E0 = 50


    nTimes = 100000

    fracasos, fracasos2 = countFracasos(nTimes, T, mu, sigma, L, E0, tau)
    print(fracasos/nTimes)

    ts_death = np.log(E0)*tau 
    p_death = 1-scipy.stats.norm(T, sigma).cdf(ts_death)
    print('ts* = ', ts_death, 'prob death', p_death)
    binomial_pmf = 1-binom.pmf(0, L, p_death)
    print(binomial_pmf)
    
    
    '''
    print('ts* = ', ts_death, 'prob death', p_death)
    print(fracasos)
    binomial_pmf = binom.pmf(1, L, p_death)
    print(binomial_pmf)
    '''
    #plt.plot(t_trajectory[1:], E_trajectory)
    #plt.axhline(y=1, c = 'r', ls = '--')
    #plt.show()


if __name__ == '__main__':
    main()
