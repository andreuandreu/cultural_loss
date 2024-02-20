import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import name_files as nf
from mpl_toolkits.axes_grid1 import make_axes_locatable




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
    # ax1.set_title('recurrence series $T = $' + "{:.1f}".format(per_in_yr) + \
    #               '[yr] $\sigma_{T} = $' + "{:.2}".format(var.false_ratio) + \
    #              ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' $\tau =$' + \
    #              "{:.1f}".format(var.tau*par.time_step) + '[yr]')
    ax1.set_title('recurrence series $T = $' + "{:.1f}".format(per_in_yr) +
                  '[yr] $\sigma_{T} = $' + "{:.2}".format(var.noiseLevel) +
                  ' $N_e = $' + "{:d}".format(var.pop) + r' $\tau =$' +
                  "{:.1f}".format(var.tau*par.time_step) + '[yr]')

    # Plot power spectrum of boolean vector
    x_scale = np.arange(2*par.vector_length/var.periode) * \
        var.periode**2/par.vector_length
    ax2.plot(x_scale, power_spectrum[0:len(x_scale)])  #
    # ax2.plot(np.arange(len(vector))*par.time_step, power_spectrum)
    ax2.set_xlabel('periode')
    ax2.set_ylabel('Power')
    ax2.set_title('Power Spectrum')

    # plt.tight_layout()




def plot_traitTime_evol(trait_series, time_series,  var, par):

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '[yr] $\sigma_{T}= $' + "{:.2f}".format(var.noiseLevel) +
    #              ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' r$\tau =$' + "{:.1f}".format(var.tau) + '[yr]')

    ax1.set_title('$T =$' + "{:.2f}".format(var.periode) +
                  '[yr] $\sigma_{T} =$' + "{:.2f}".format(var.noiseLevel) +
                  ' $N_e =$' + "{:d}".format(var.pop) +
                  r' $\tau =$' + "{:.1f}".format(var.tau*par.time_step) + '[yr]')

    # Plot time series of boolean vector
    ax1.plot(np.arange(len(trait_series))*par.time_step, trait_series)

    ax1.hlines(y=0, xmin=0, xmax=len(trait_series)*par.time_step,
               ls='--', linewidth=1.2, color='k')
    # ax1.set_xlabel('time')
    ax1.set_ylabel('$N_e(t)/N^0_e$')

    # Plot power spectrum of boolean vector
    ax2.plot(np.arange(len(trait_series))*par.time_step, time_series)  #
    ax2.set_xlabel('time')
    ax2.set_ylabel('$\Delta t$')

    plt.tight_layout()


def plot_traitTime_evol_and_noise_sequence(trait_series, noisy_dependence, var, par):

    # Create figure with two subplots
    fs = 18
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,  gridspec_kw=dict(hspace=0))

    # ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '[yr] $\sigma_{T}= $' + "{:.2f}".format(var.noiseLevel) +
    #              ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' r$\tau =$' + "{:.1f}".format(var.tau) + '[yr]')

    ax1.set_title('$T =$' + "{:.2f}".format(var.periode) +
                  '[yr] $\sigma_{T} =$' + "{:.2f}".format(var.noiseLevel) +
                  ' $N_e =$' + "{:d}".format(var.pop) +
                  r' $\tau =$' + "{:.1f}".format(var.tau*par.time_step) + '[yr]')

    # Plot time series of boolean vector
    ax1.plot(np.arange(len(trait_series))*par.time_step, trait_series)  #

    ax1.hlines(y=0, xmin=0, xmax=len(trait_series)*par.time_step,
               ls='--', linewidth=0.8, color='k')
    # ax1.set_xlabel('time')
    ax1.set_ylabel('$N_e(t)/N^0_e$')

    # Plot event sequence

    for i,l in enumerate(noisy_dependence[:len(trait_series)]):
        if l == 1:
            ax2.vlines(x=i*par.time_step, ymin=0, ymax=1,
                ls='-', linewidth=1.1, color='r')
    
    #ax2.plot(np.arange(len(trait_series))*par.time_step,
    #    noisy_dependence[:len(trait_series)])  # 
    ax2.set_xlabel('Time [yr]')
    ax2.set_ylabel('Recurrent performance')

    # nf.name_survival_fig(to_modify, par.dropbox_dir+'.svg', var, par)
    surv_time = str(int(len(trait_series)*par.time_step)) + 'yrs'
    name_fig = par.dropbox_dir + '/plots/fig_1_sequence/fig_1_sequence_'+ surv_time

    print('FFFFFFF', name_fig)
    plt.savefig(name_fig+'.svg', bbox_inches='tight')
    plt.savefig(name_fig+'.png', bbox_inches='tight')
    plt.savefig(name_fig+'.eps', bbox_inches='tight')

    plt.tight_layout()


def plot_multiple_traitTime_evol(ax1, ax2, trait_series, time_series, var, par, alpha, lw=0.3):

    # Plot time series of boolean vector
    # ax1.set_title('$T= $' + "{:.2f}".format(var.periode) + '$[t_s]$ $\sigma_{T}= $' + "{:.2f}".format(var.noiseLevel) +
    #              ' $k_{\epsilon} = $' + "{:.2}".format(var.kError) + ' $t12 =$' + "{:.1f}".format(var.tau))

    ax1.set_title('$T= $' + "{:.2f}".format(var.periode) +
                  '$[t_s]$ $\sigma_{T}= $' + "{:.2f}".format(var.noiseLevel) +
                  ' $N_e = $' + "{:d}".format(var.pop) +
                  ' r$\tau =$' + "{:.1f}".format(var.tau*par.time_step))

    ax1.plot(np.arange(len(trait_series))*par.time_step,
             trait_series, color='orange', lw=lw, alpha=alpha)
    # ax1.set_xlabel('step')
    ax1.set_ylabel('$N_e(t)/E^0_p$')

    # Plot power spectrum of boolean vector
    ax2.plot(np.arange(len(trait_series))*par.time_step,
             time_series,  color='blue', lw=lw, alpha=alpha)  #
    ax2.set_xlabel('time')
    ax2.set_ylabel('$\Delta t$')

    # plt.tight_layout()

    # Print amplitude at periode of interest
    # print(f"Amplitude at {signal_periode} Hz: {np.abs(dft[freq_index])}")

def plot_multiple_noiseRealizations(k_series_set, Dt_series_set, len_series_set, var, par, hist_bins):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    var.kError = nf.which_threshlod(var, par)
    # ax1.hlines(y=var.kError, xmin=0, xmax=vector_length*par.time_step,
    #           ls='--', linewidth=1.2, color='r')
    ax1.hlines(y=1/var.pop, xmin=0, xmax=vector_length*par.time_step,
               ls='--', linewidth=1.2, color='r')

    noise_range = np.arange(0, 0.5, 0.5/par.Nrealizations)
    for i in range(par.Nrealizations):
        alpha = noise_range[i]
        plot_multiple_traitTime_evol(
            ax1, ax2, k_series_set[i], Dt_series_set[i], var, par, alpha)

    # Flatten the array into a 1D array
    data = np.array(len_series_set).flatten()
    # print('dadada', data)
    ax3.hist(data, bins=hist_bins)
    ax3.set_xlabel('len of serie')
    ax3.set_ylabel('periode')


def plot_Period_tau_dep(var, par):
    fig, ax = plt.subplots()

    a1 = 1/np.logspace(0, 3, 4)
    a2 = np.arange(0.001, -np.log(var.kError), 1)
    log_tau_range = np.outer(a1, a2).flatten()*par.time_step
    print('taus', log_tau_range*par.time_step)
    Tmax = - np.log(var.kError)/log_tau_range
    # print('Tmaxssss', Tmax)
    ax.scatter(log_tau_range, Tmax)
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_threshold_functions_preiode(var, par):
    fig, ax = plt.subplots()

    threshold1 = np.exp(-1/(var.periodes))
    threshold2 = np.exp(-1/(var.periodes*par.time_step))
    threshold05 = np.exp(-par.time_step/(var.periodes))

    # ax.plot(periodes, threshold1, c='y', label ='exp(-1/T)')
    # ax.plot(periodes, threshold2, c='g',  label='exp(-1/(T*ts))')
    # ax.plot(periodes, threshold05, c='r', label='exp(-ts/T)')

    # ax.plot(periodes, 1-threshold1, c='m', label='1-exp(-1/T)')
    ax.plot(var.periodes, 1-threshold2, c='b',  label='$1-e^{-1/T}$')
    # ax.plot(periodes, 1-threshold05, c='c', label='1-exp(-ts/T)')

    ax.legend(frameon=False)
    ax.set_ylabel('$k_{\epsilon}$')
    ax.set_xlabel('$T$')


def plot_threshold_functions_tau(var, par):
    fig, ax = plt.subplots()

    threshold1 = 2**(-1/(var.taus))
    threshold2 = 1-2**(-1/(var.taus*par.time_step))
    threshold05 = 2**(-par.time_step/(var.taus))

    # ax.plot(periodes, threshold1, c='y', label ='exp(-1/T)')
    # ax.plot(periodes, threshold2, c='g',  label='exp(-1/(T*ts))')
    # ax.plot(periodes, threshold05, c='r', label='exp(-ts/T)')

    # ax.plot(periodes, 1-threshold1, c='m', label='1-exp(-1/T)')
    ax.plot(var.taus, threshold2, c='b',  label=r'$1-2^{-1/\tau}$')
    # ax.plot(periodes, 1-threshold05, c='c', label='1-exp(-ts/T)')

    ax.legend(frameon=False)
    ax.set_ylabel('$k_{\epsilon}$')
    ax.set_xlabel(r'$\tau$')



def plot_threshold_function(var, par):

    threshold_mat = []
    for h in var.taus:
        threshold_row = []
        var.tau = h
        for l in var.periodes:
            var.periode = l
            threshold = kd.which_threshlod(var, par)
            threshold_row.append(threshold)
        threshold_mat.append(threshold_row)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    # n_max_mat = np.array(threshold_mat)
    # aux = np.where(threshold_mat > 86)
    # n_max_mat[aux] = 0
    # aux = np.where(threshold_mat < 2)
    # n_max_mat[aux] = 86

    # im = ax.imshow(n_max_mat, cmap='bone')  # extent = extent, 'bone'
    im = ax.contourf(var.periodes, var.taus, threshold_mat,
                     extend="both", cmap='bone')

    tagX, labelsX = nf.var_tagAndLabels('periode', var.periodes)
    tagY, labelsY = nf.var_tagAndLabels('tau', var.taus)

    # ax.set(xticks=np.arange(len(var.periodes)), xticklabels=labelsX,
    #       yticks=np.arange(len(var.taus)), yticklabels=labelsY)

    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)

    fig.colorbar(im, cax=cax, orientation='vertical')

    return threshold_mat





def main():
    if __name__ == '__main__':
        main()