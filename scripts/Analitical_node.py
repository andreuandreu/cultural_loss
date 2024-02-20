import matplotlib.pyplot as plt
import matplotlib
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
    E_values = E0 * np.exp(-time_array/tau)

    if E_values[-1] < 1:
        i = np.where(E_values < 1)
        E_values = E_values[0:i[0][0]]
        E_values[-1] = 0
        time_array = time_array[0:i[0][0]]
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

            if time_array_serie[-1] > 1000:
                break


    aux = np.where(E_values_serie == E0)
    events = np.zeros(len(E_values_serie))
    events[aux] = 1

    return fracasos, E_values_serie[1:], events[1:], time_array_serie[1:]


def generateDecaySeries(eventsTimes, L2, E0, tau):
    fracasos = 0

    E_values_serie = [E0]
    for t in  eventsTimes:
        E_values, t_values = build_train_pulses(t, E0, tau, L2)
        E_values_serie = np.append(E_values_serie, E_values) 

        if len(E_values_serie) > 2000:
            break
    
    return E_values_serie


@jit((int64)(int64, float64, float64, float64, int64, float64, int64), nopython=True, fastmath=True)
def countTempFracasos(nTimes, T, mu, sigma, Nper, t_death, timeLim):
    fracasos = 0
    timeLength = 0
    for i in range(0, nTimes):
        ts_dist =  np.random.normal(T, sigma, Nper)
        for ts in ts_dist:
            timeLength =+ ts
            if ts >= t_death:
                fracasos += 1
                break
            if timeLength > timeLim:
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


@jit((int64)(int64, int64, float64, float64, float64), nopython=True, fastmath=True)
def countPeriodFailures(nTimes, L, T, noise_tolerance, t_death):
    fracasos = 0

    Nper = int(L/T)
    for i in range(0, nTimes):
        count_miss = 0
        for j in range(0, Nper+T):
            miss = np.random.random(1)

            if T*j > L-T:
                break  
            elif T >  t_death:
                fracasos += 1
                break
            elif miss < noise_tolerance:
                count_miss =  count_miss + 1
                #print('mimimi', j, count_miss)
                if count_miss*T > t_death:
                    fracasos += 1
                    break
            else:
                count_miss = 0
            
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
    
    return fracasos

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)


    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", )

    # Show all ticks and label them with the respective list entries.
    col_labels_str = []
    for i, e in enumerate(col_labels):
        col_labels_str = np.append(col_labels_str,f"{e:.0f}")

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels_str )
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.0f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



def tdeathMat(E0s, taus):


    t_deathMat = np.log(E0s[:, np.newaxis]) * taus[np.newaxis, :]

    return t_deathMat 




def plot_tDeathMat(E0_max, tau_max):
   
    E0s = np.arange(2,E0_max, 4)
    taus = np.arange(1, tau_max, (tau_max)/len(E0s))
    t_deathMat  = tdeathMat(E0s, taus)

    fig, ax = plt.subplots()
    #ax.matshow(t_deathMat, cmap=plt.cm.Blues)


    im, cbar = heatmap( t_deathMat, E0s, taus, ax=ax,
                    cmap="YlGn", cbarlabel="$\Delta t_{th} [yr]$")#"{taus:.0f}"
    
    
    valmt = {'float_kind':lambda x: "%.0f" % x}#np.array2string(t_deathMat, precision=1)#f"{t_deathMat:.1f}"
    texts = annotate_heatmap(im, data=t_deathMat, valfmt="{x:.0f}")
 


    #im = ax.imshow(t_deathMat, cmap=plt.cm.Blues)

    # Show all ticks and label them with the respective list entries
    #ax.set_xticks(np.arange(len(taus)), labels=taus)
    #ax.set_yticks(np.arange(len(E0s)), labels=E0s)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), ha="right")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(taus)-1):
    #    for j in range(len(E0s)-1):
    #        text = ax.text(j, i, int(t_deathMat[i, j]),
    #                    ha="center", va="center", color="w")

    #ax.set_title("threshold matrix")
    #fig.tight_layout()

    ax.set_ylabel('$N_e(t)$', fontsize = 10)
    ax.set_xlabel(r'$\tau $[yr]', fontsize = 10)
    
    #plt.annotate('$\Delta t_{th}$', xy=(t_death/2, E0-3), fontsize = 13)
    


def plot_exp_dec(tau, E0):
   

    t_death = np.log(E0) * tau

    time = np.arange(int(t_death+1.5))

    exp = E0*np.exp(-time/tau)

    exp = np.append(exp, 0)
    time = np.append(time, time[-1]+1)

    fig, ax = plt.subplots()
    plt.annotate('$\Delta t_{th}$', xy=(t_death/2, E0-3), fontsize = 13)
    plt.arrow(0, E0, t_death, 0,  linewidth=2,  head_width=1.05, head_length=1.03, color='k', length_includes_head=True)#t_death,
    plt.arrow(t_death, E0, -t_death, 0,  linewidth=2,  head_width=1.05, head_length=1.03, color='k', length_includes_head=True)#t_death,


    ax.vlines(x=t_death, ymin=0, ymax=E0, ls='-', linewidth=1.8, color='r')
    ax.hlines(y=1, xmin=0, xmax=t_death+1, ls='--', linewidth=1.5, color='k')
    ax.plot(time, exp,  linewidth=1.8)
    #plt.hist(distances, bins=11)
    ax.set_ylabel('$N_e(t)$', fontsize = 10)
    ax.set_xlabel('Time [yr]', fontsize = 10)


def plot_reconstructed_sequence(t_series, E_series, events, periode, noiseLevel, pop, halfLife):
     
    fs = 18
    fig, ax1 = plt.subplots(1, 1)

    name_fig = ''

    ax1.set_title('$T =$' + "{:.2f}".format(periode) +
                  '[yr] $\sigma_{T} =$' + "{:.0f}".format(noiseLevel*100) + '%' +
                  ' $N_e =$' + "{:d}".format(pop) +
                  r' $\tau =$' + "{:.1f}".format(halfLife) + '[yr]')
    #ax.plot(ts_dist_pos[1:], E_series[1:]) 
    ax1.plot(t_series, E_series) 
    ax1.plot(t_series, events)
    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('Recurrent performance')

    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('Events & $N_e(t)$')

    #plt.savefig(name_fig+'.svg', bbox_inches='tight')
    #plt.savefig(name_fig+'.png', bbox_inches='tight')
    #plt.savefig(name_fig+'.eps', bbox_inches='tight')
    


def main():

    mu = 0
    T = 4
    L = 1000
    L2 = 100
    sigma = 1.5 * T
    noise_tolerance = 0.61
   
    Nper = int(L/T)

    tau = 4

    E0 = 44
    E0_max = E0
    tau_max = 25
    plot_exp_dec(tau, E0)


    t_death = np.log(E0) * tau

    plot_tDeathMat(E0_max, tau_max)

    nTimes = 1222

    failures = countPeriodFailures( nTimes,  Nper, T, noise_tolerance, t_death) 
    print('\nmisses    ', failures/nTimes, 'Surb', 1-failures/nTimes, '\n')

    max_falses = 0.9  # 0.6#0.66
    min_falses = 0.01
    step_falses = (max_falses - min_falses)/11.
    falses_ratios = np.arange(min_falses, max_falses, step_falses)

    for e in falses_ratios:
        failures = countPeriodFailures( nTimes,  Nper, T, e, t_death) 
        print('misses    ', e, failures/nTimes, 'Surb', 1-failures/nTimes)

    '''simulated options'''
    fracasos, E_series, events, t_series = countFracasos(nTimes, T, mu, sigma, Nper, L2, E0, tau)
    fracasosTemp =    countTempFracasos(nTimes, T, mu, sigma, Nper, t_death, L)
    fracasosPosTemp = countTempPositiveFracasos(nTimes, T, mu, sigma, Nper, t_death, L)
    fracasosVar = create_noisy_period_series_var(nTimes, L, sigma, t_death)
    fracasosPer =     create_noisy_period_series(nTimes, L, T, sigma, t_death)
    
    #E_series = generateDecaySeries(ts_dist_pos, L2, E0, tau)
    #print('sim Frac', fracasos/nTimes, 'Surb', 1-fracasos/nTimes)
    print('\nsimTim    ', fracasosTemp/nTimes, 'Surb', 1-fracasosTemp/nTimes)
    print('simPosTim ', fracasosPosTemp/nTimes, 'Surb', 1-fracasosPosTemp/nTimes)
    print('sim Var    ', fracasosVar/nTimes, 'Surb', 1-fracasosVar/nTimes)
    print('sim Per  ', fracasosPer/nTimes, 'Surb', 1-fracasosPer/nTimes, '\n')

    plot_reconstructed_sequence(t_series, E_series, events, T, sigma/T, E0, tau)
    
    p_surb = scipy.stats.norm(T, sigma).cdf(t_death)
    p_surb_po = scipy.stats.poisson.cdf(T, sigma)
    p_death = 1- p_surb

    print('ts* = ', t_death, 'prob surb', p_surb)
    #binomial_pmf = 1-binom.pmf(0, Nper, p_death)
    cumulat_p_surb = p_surb**Nper
    #cumulat_p_death = (1-p_death)**Nper
    cumulat_p_surb_po = p_surb_po**L
    #print(binomial_pmf)
    print('\nanaly norm ', 1-cumulat_p_surb, 'surb', cumulat_p_surb)
    #print('analy binom ', 1-cumulat_p_death,'Surb', cumulat_p_death)
    print('analy norm     ', 1-cumulat_p_surb_po, cumulat_p_surb_po,'\n')



    #fracasos_fail = create_stocastic_dependence(nTimes, false_ratio, T, L, t_death) 
    

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
