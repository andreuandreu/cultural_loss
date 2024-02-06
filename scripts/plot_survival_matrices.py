import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import name_files as nf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import knode_decay as kd
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.ticker import MaxNLocator


def plot_analy_survival_matrix(varNameY, valuesY, varNameX, valuesX, name_mat, var, par, analy=False):

    fig, ax = plt.subplots()
    fs = 14

    analy_mat = np.load(name_mat)

    im = ax.pcolormesh(analy_mat,  cmap='OrRd')

    
  
    fig.text(0.95, 0.5, r"$N_e(1Kyr)/N_e(0)$", va="center", rotation=-90, fontsize=fs)

    print('xxxxxxxx', valuesX)
    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    ax.set_ylabel(tagY, fontsize = fs)
    ax.set_xlabel(tagX, fontsize = fs)


    # Format tick labels to display only one decimal place
    ax.set_xticks(np.arange(len(valuesX)))
    labels_of_interest = []  # [str(i) for i in xLavels]
    for i, l in enumerate(valuesX):
        if i%5 == 2:
            labels_of_interest = np.append(labels_of_interest, f"{l + valuesX[0]:.0f}")
        else: 
            labels_of_interest = np.append(labels_of_interest, '')

    ax.set_xticklabels(labels_of_interest, fontsize=fs-1)
    #ax.set_xticklabels(xLavels, fontsize=fs-1)

    # Format tick labels to display only one decimal place and apear once every 5 values
    ax.set_yticks(np.arange(len(valuesY)))
    labels_of_interest = []
    for i, l in enumerate(valuesY):
        if i % 8 == 2:
            labels_of_interest = np.append(
                labels_of_interest, f"{100*(l)- valuesY[1]:.0f}")
        else:
            labels_of_interest = np.append(labels_of_interest, '')
    ax.set_yticklabels(labels_of_interest, fontsize=fs-1)

    ax.tick_params(width=0, length=0)

    #ax.set_title(title)
    #x_locator = MultipleLocator(base=12)
    #y_locator = MultipleLocator(base=8)
    #ax.xaxis.set_major_locator(x_locator)
    #ax.yaxis.set_major_locator(y_locator)  

    #plt.yticks()

    


    #ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    #ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    to_modify = varNameX + '_' + varNameY
    name_fig = nf.name_survival_fig(
        to_modify, par.plots_survivalMat, var, par, analy)





def colum_multiplot_analy_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, analy, var, par, rows=1):

    # create a figure and set the size
    cols = len(valuesZ)
    # fig, axs = plt.subplots(rows, cols, sharey=True, subplot_kw=dict(
    #    frameon=False))  # sharex=True, sharey=True

    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    # [['0', '1', '2', '3', '4', '5']]
    mossaic_keys = [np.arange(len(valuesZ)).astype(str)]
    print('mmomomo', mossaic_keys)

    hight = 12.1
    width = 12.1
    fig, axs = plt.subplot_mosaic(
        mossaic_keys,
        sharex=True,
        sharey=True,
        figsize=(width, hight),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    labelsX_short = []
    intervalX = 3
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])

    # for i in range(rows):
    for j in range(cols):
        name = nf.file_name_n_varValue(varNameZ, valuesZ[j], var, par)
        to_modify = varNameX + '_' + varNameY
        name_mat = nf.name_survival_ranges(to_modify, 'mat_', var, par, analy) + '.npy'
        analy_mat = np.load(name_mat)
        im = axs[str(j)].pcolormesh(analy_mat,  cmap='OrRd')

def plot_survival_martrix(varNameY, valuesY, varNameX, valuesX, var, par, analy=''):

    fig, ax = plt.subplots()
    survival_rate = np.empty([len(valuesY), len(valuesX)])
    for i, valY in enumerate(valuesY):
        nameY = nf.file_name_n_varValue(varNameY, valY,  var, par, analy)
        for j, valX in enumerate(valuesX):
            # print('noise', "{:.2f}".format(var.false_ratio), 'per', "{:.2f}".format(var.periode))

            nameX = nf.file_name_n_varValue(varNameX, valX,  var, par, analy)
            dataset_len_series = np.load(nameX)
            # print('ufufufuf', dataset_len_series)
            survivors = len(np.where(dataset_len_series >
                            par.vector_length-10)[0])
            survival_rate[i][j] = survivors/par.Nrealizations
            # print('sisisiusususu', survivors/par.Nrealizations)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    x, y = np.meshgrid(var.periodes, var.halfLifes)
    # extent = extent, 'bone'
    #im = ax.imshowh(survival_rate,   extent=[
    #               x.min(), x.max(), y.max(), y.min()], cmap='OrRd')
    
    print('mamamama', survival_rate)
    im = ax.pcolormesh(survival_rate, cmap='OrRd')
    # /np.log(100*var.noiseLevel)
    # ax.plot(var.periodes, -var.periodes/np.log2(1/(var.pop-1)))
    # ax.plot(var.periodes, -var.periodes/np.log(1/(var.pop-1)))
    # ax.plot(var.periodes, var.periodes/np.log(2))
    # for e in var.noiseLevels:
    #    ax.plot(var.periodes, var.halfLifes*e/np.log(2))
    # ax.plot(var.periode, var.halfLife)
    # ax.plot(var.periode, var.halfLife*var.noiseLevel)
    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    # ax.set(xticks=np.arange(len(valuesX)), xticklabels=labelsX,
    #       yticks=np.arange(len(valuesY)), yticklabels=labelsY)

    # title = r'$\tau$ = ' + "{:4.0f}".format(var.halfLife*par.time_step) + '[yr]'
    title = '$\sigma_{T}$ = ' + \
        "{:3.3f}".format(var.noiseLevel) + '[%]'
    ax.set_ylabel(tagY)
    ax.set_xlabel(tagX)
    ax.set_title(title)

    fig.colorbar(im, cax=cax, orientation='vertical')

    to_modify = varNameX + '_' + varNameY
    name_fig = nf.name_survival_fig(
        to_modify, par.plots_survivalMat, var, par, analy)

    # plot(ax1, V.real)

    # ax.plot(par.vector_length *
    #        0.005/(var.periodes*par.time_step)**2)

    print('nananannana', name_fig)
    # plt.savefig(name_fig, bbox_inches='tight')


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
                name = nf.file_name_n_varValue(varName, var.peridodes[j])
                dataset_len_series = np.load(name)
                # print('ufufufuf', dataset_len_series)

                # axs.set_title('T=' + "{:.2f}".format(var.periodes[j]))
                axs[i][j].hist(dataset_len_series, bins=hist_bins)
            l += 1


def multiplot_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, var, par, rows=1):

    # create a figure and set the size
    cols = len(valuesZ)
    # fig, axs = plt.subplots(rows, cols, sharey=True, subplot_kw=dict(
    #    frameon=False))  # sharex=True, sharey=True

    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    # [['0', '1', '2', '3', '4', '5']]
    mossaic_keys = [np.arange(len(valuesZ)).astype(str)]
    print('mmomomo', mossaic_keys)

    hight = 12.1
    width = 12.1
    fig, axs = plt.subplot_mosaic(
        mossaic_keys,
        sharex=True,
        sharey=True,
        figsize=(width, hight),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    labelsX_short = []
    intervalX = 3
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])

    # for i in range(rows):
    for j in range(cols):
        name = nf.file_name_n_varValue(varNameZ, valuesZ[j], var, par)
        # if varNameZ == 'halfLife':
        #    var.halfLife = valuesZ[j]

        survival_rate = kd.a_survival_martrix(
            varNameY, valuesY, varNameX, valuesX, var, par)
        im = axs[str(j)].imshow(survival_rate, cmap='OrRd')
        if j == 0:
            axs[str(j)].set_ylabel(tagY)
            # axs[str(j)].set(xticks=np.arange(0, len(valuesZ), intervalX), xticklabels=labelsX_short,
            #      yticks=np.arange(len(valuesY)), yticklabels=labelsY)
        # if j > 0:
            # axs[str(j)] = axs[j-1].twiny()
            # axs[j].set(xticks=np.arange(len(valuesX)), xticklabels=labelsX)
        title = nf.set_title_mat(varNameX, varNameY, j, cols, var, par)
        # axs[str(j)].set_xlabel(tagX)
        axs[str(j)].set_title(title)


def prepare_mxn_figure(fig, rows, valuesX, valuesY, valuesS):

    left = 0.01
    width = 0.7
    bottom = 0.01
    height = 0.8
    right = left + width
    top = bottom + height

    fs = 11

    fig.text(0.92, 0.89, "$N_e$", va="center", fontsize=fs)

    poss = np.arange(0.75, 0, -1/(rows+2))

    for p, v in zip(poss, valuesS):
        #print('whaaaatTTTTT???', p, v)
        fig.text(0.92, p, str(v), va="center", fontsize=fs)

    textX = 'T ='
    for v in valuesX:
        textX = textX + "{:.1f}".format(v) + '   '

    textY = '\sigma_{T} ='
    for v in valuesY:
        textY = textY + "{:.1f}".format(v) + '   '

    fig.text(0.5, 0.02, r"$T$[yr]", va="center", fontsize=fs)
    fig.text(0.02, 0.5, r"$\sigma_{T}$ [%]", va="center", fontsize=fs)


def multiplot_mxn_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, varNameS, valuesS, var, par):

    # create a figure and set the size
    cols = len(valuesZ)
    rows = len(valuesS)
    # row and column sharing
    f, axs = plt.subplots(len(valuesS), len(valuesZ),
                          sharex=True, sharey=True, gridspec_kw=dict(hspace=0))
    f.subplots_adjust(wspace=0, hspace=0)

    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    labelsX_short = []
    intervalX = 3
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])
    
    count = 0
    for i in range(rows):
        name = nf.file_name_n_varValue(varNameS, valuesS[i])
        for j in range(cols):
            name = nf.file_name_n_varValue(varNameZ, valuesZ[j])
            survival_rate = kd.a_survival_martrix(
                varNameY, valuesY, varNameX, valuesX)
            print('riiiight?  . ', count)
            im = axs[i, j].pcolormesh(survival_rate,  cmap='OrRd')

            if i == 0:
                title = nf.set_title_mat(varNameX, varNameY, j, cols)
                axs[i, j].set_title(title)
            count += 1
    prepare_mxn_figure(f, rows, valuesX, valuesY, valuesS)


def multiplot_mxn_analy_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, varNameS, valuesS, var, par):

    # create a figure and set the size
    cols = len(valuesZ)
    rows = len(valuesS)
    # row and column sharing
    f, axs = plt.subplots(len(valuesS), len(valuesZ),
                          sharex=True, sharey=True, gridspec_kw=dict(hspace=0))
    f.subplots_adjust(wspace=0, hspace=0)

    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    labelsX_short = []
    intervalX = 3
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])
    to_modify = varNameX + '_' + varNameY
    count = 0
    # Remove ticks from both x and y axes


    for i in range(rows):
        name = nf.file_name_n_varValue(varNameS, valuesS[i], var, par)
        for j in range(cols):
            name = nf.file_name_n_varValue(varNameZ, valuesZ[j], var, par) 
            name_mat = nf.name_survival_ranges(
                to_modify, 'mat_', var, par, 'analy') 
            analy_mat = np.load(name_mat+'.npy')
            im = axs[i,j].pcolormesh(analy_mat,  cmap='OrRd')

            if i == 0:
                title = nf.set_title_mat(varNameX, varNameY, j, cols, var, par)
                axs[i, j].set_title(title)
            
            count += 1
    plt.xticks([])
    plt.yticks([])
    prepare_mxn_figure(f, rows, valuesX, valuesY, valuesS)

    name_fig = par.dropbox_dir + '/plots/fig3_mxn_mat/fig3-2_analy_' + \
        str(len(var.periodes)) + 'x' + str(len(var.noiseLevels))
    #name_fig = nf.name_survival_fig(
    #    'pop_periode_halfLife_noiseLevel', par.dropbox_dir, var, par, 'analy')
    plt.savefig(name_fig+'.svg', bbox_inches='tight')
    plt.savefig(name_fig+'.png', bbox_inches='tight')
    plt.savefig(name_fig+'.eps', bbox_inches='tight')


def multiplot_mxn_alber_survivals(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, varNameS, valuesS, var, par):

    # create a figure and set the size
    cols = len(valuesZ)
    rows = len(valuesS)
    # row and column sharing
    f, axs = plt.subplots(len(valuesS), len(valuesZ),
                          sharex=True, sharey=True, gridspec_kw=dict(hspace=0))
    f.subplots_adjust(wspace=0, hspace=0)

    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    labelsX_short = []
    intervalX = 3
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])
    to_modify = varNameX + '_' + varNameY
    count = 0
    # Remove ticks from both x and y axes

    for i in range(rows):
        name = nf.file_name_n_varValue(varNameS, valuesS[i], var, par)
        for j in range(cols):
            name = nf.file_name_n_varValue(varNameZ, valuesZ[j], var, par)
            name_mat = nf.name_survival_ranges(
                to_modify, 'mat_', var, par, 'alber')
            analy_mat = np.load(name_mat+'.npy')
            im = axs[i, j].pcolormesh(analy_mat,  cmap='OrRd')

            if i == 0:
                title = nf.set_title_mat(varNameX, varNameY, j, cols, var, par)
                axs[i, j].set_title(title)

            count += 1
    plt.xticks([])
    plt.yticks([])
    prepare_mxn_figure(f, rows, valuesX, valuesY, valuesS)

    name_fig = par.dropbox_dir + '/plots/fig3_mxn_mat/fig3-2_alber_' + \
        str(len(var.periodes)) + 'x'+ str(len(var.noiseLevels))
    #name_fig = nf.name_survival_fig(
    #    'pop_periode_halfLife_noiseLevel', par.dropbox_dir, var, par, 'alber')
    plt.savefig(name_fig+'.svg', bbox_inches='tight')
    plt.savefig(name_fig+'.png', bbox_inches='tight')
    plt.savefig(name_fig+'.eps', bbox_inches='tight')


def multiplot_survivals_seaborn(varNameY, valuesY, varNameX, valuesX, varNameZ, valuesZ, var, par, rows=1):

    # create a figure and set the size
    cols = len(valuesZ)
    # fig, axs = plt.subplots(rows, cols, sharey=True, subplot_kw=dict(
    #    frameon=False))  # sharex=True, sharey=True
    fig, axs = plt.subplots(ncols=cols)

    tagX, labelsX = nf.var_tagAndLabels(varNameX, valuesX, var, par)
    tagY, labelsY = nf.var_tagAndLabels(varNameY, valuesY, var, par)

    labelsX_short = []
    intervalX = 3
    for i in range(len(labelsX)):
        if i % intervalX == 0:
            labelsX_short.append(labelsX[i])

    surv_matrices = np.array([])  # .reshape(len(valuesY), len(valuesZ))

    for v in valuesZ:

        name = nf.file_name_n_varValue(varNameZ, v)

        survival_rate = kd.a_survival_martrix(
            varNameY, valuesY, varNameX, valuesX)

        # surv_matrices[j] = survival_rate
        surv_matrices = np.concatenate((surv_matrices, survival_rate))

    df = pd.DataFrame(data=surv_matrices, columns=valuesZ)
    sns.heatmap(df)


def plot_max_death_interval(var, par):

    n_max_mat = []
    for e in var.kErrors:
        n_max_row = []
        for l in var.halfLifes:
            n_max_row.append(par.time_step * np.log(var.k0/e)/l)
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

    tagX, labelsX = nf.var_tagAndLabels('halfLife', var.halfLifes, var, par)
    tagY, labelsY = nf.var_tagAndLabels('kError', var.kErrors, var, par)

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





