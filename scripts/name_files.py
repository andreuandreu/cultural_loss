import os



def file_name_n_varValue(to_modify, value, var, par, analy = ''):

    path = par.output_dir + par.root + '_ts='+str(par.time_step) + '_L=' + str(par.vector_length) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    if to_modify == 'periode':
        var.periode = value
        
    elif to_modify == 'tau':
        var.tau = value
        
    elif to_modify == 'false_ratio':
        var.false_ratio = value

    elif to_modify == 'noiseLevel':
        var.noiseLevel = value
        
    elif to_modify == 'kError':
        var.kError = value

    elif to_modify == 'pop':
        var.pop = value

    elif to_modify == 'tDeath':
        var.tDeath = value
        name = path + analy +\
        'T='   + "{:.2f}".format(var.periode) + \
        '_Te=' + "{:.2f}".format(var.noiseLevel) + \
        '_tth='+ "{:.1f}".format(var.tau) + '.npy'
        return name
        
    else:
        print('wrong to modify name argument!!! be careful, only options are periode, tau, false_ratio, kError, pop, tDeath')
        quit()
   

    name = path + analy +\
    'T='   + "{:.2f}".format(var.periode) + \
    '_Te=' + "{:.2f}".format(var.noiseLevel) + \
    '_t12='+ "{:.1f}".format(var.tau)+ \
    '_Ep=' + "{:d}".format(var.pop) + '.npy'  # + str(var.kError)
    #print('HELOOO????', name)
    return name


def name_survival_fig(to_modify, root_fig, var, par, analy=''):

    if 'periode' in to_modify:
        periode_seg = '_Tran-' + \
            "{:.2f}".format(var.periodes[0])+'-' + \
            "{:.2f}".format(var.periodes[-1])
    else:
        periode_seg = '_T-' + "{:.2f}".format(var.periode)

    if 'tau' in periode_seg:
        tau_seg = '_t12ran-' + \
            "{:.1f}".format(var.taus[0]) + \
            '-'+"{:.1f}".format(var.taus[-1])
    else:
        tau_seg = '_t12-' + "{:.1f}".format(var.tau)

    if 'false_ratio' in to_modify:
        false_ratio_seg = '_FRran-' + \
            "{:.2f}".format(
                var.falses_ratios[0])+' -'+"{:.2f}".format(var.falses_ratios[-1])
    else:
        false_ratio_seg = '_FR-' + "{:.2f}".format(var.false_ratio)

    if 'noiseLevel' in to_modify:
        noiseLevel_seg = '_TNran-' + \
            "{:.2f}".format(var.noiseLevels[0]) + \
            '-'+"{:.2f}".format(var.noiseLevels[-1])
    else:
        noiseLevel_seg = '_TN-' + "{:.2f}".format(var.noiseLevel)

    if 'kError' in to_modify:
        kError_seg = '_kEran-' + \
            "{:.2f}".format(var.kErrors[0])+' -' + \
            "{:.2f}".format(var.kErrors[-1])
    else:
        kError_seg = '_kE-' + "{:.2f}".format(var.kError)

    if 'pop' in to_modify:
        pop_seg = '_popRan-' + \
            "{:0f}".format(var.pops[0])+' -' + \
            "{:0f}".format(var.pops[-1])
    else:
        pop_seg = '_pop-' + "{:d}".format(var.pop)

    if analy == '':
        path = par.plots_dir + root_fig + par.root +\
            '_ts=' + str(par.time_step) +\
            '_L='+str(par.vector_length) + \
            '_N=' + str(par.Nrealizations) + '/'
    else:
        path = par.plots_dir + root_fig + par.root +\
            '_' + analy + \
            '_ts=' + str(par.time_step) +\
            '_L='+str(par.vector_length) 


    if not os.path.exists(path):
        os.makedirs(path)

    name_fig = path + 'fig' + periode_seg + \
        noiseLevel_seg + tau_seg   # + kError_seg  +
    print('nonanonaonaoa', name_fig)

    return name_fig



def name_survival_ranges(to_modify, root_fig, var, par, analy = ''):
    
    if 'periode' in to_modify:
        periode_seg = '_Tran-'+"{:.2f}".format(var.periodes[0])+'-'+"{:.2f}".format(var.periodes[-1])
    else:
        periode_seg = '_T-' + "{:.2f}".format(var.periode)

    if 'tau' in periode_seg:
        tau_seg = '_t12ran-'+"{:.1f}".format(var.taus[0])+'-'+"{:.1f}".format(var.taus[-1])
    else:
        tau_seg = '_t12-' + "{:.1f}".format(var.tau)

    if 'noiseLevel' in to_modify:
        noiseLevel_seg = '_TNran-'+"{:.2f}".format(var.noiseLevels[0])+'-'+"{:.2f}".format(var.noiseLevels[-1])
    else:
        noiseLevel_seg = '_TN-' + "{:.2f}".format(var.noiseLevel)

    if  'kError' in to_modify:
        kError_seg = '_kEran-'+"{:.2f}".format(var.kErrors[0])+' -'+"{:.2f}".format(var.kErrors[-1])
    else:
        kError_seg = '_kE-' + "{:.2f}".format(var.kError)

    if 'pop' in to_modify:
        pop_seg = '_popRan-' + \
            "{:d}".format(var.pops[0])+' -' + \
            "{:d}".format(var.pops[-1])
    else:
        pop_seg = '_pop-' + "{:d}".format(var.pop)

    

    if analy == '':
        path = par.plots_dir + root_fig + par.root +\
            '_ts=' + str(par.time_step) +\
            '_L='+str(par.vector_length) + \
            '_N=' + str(par.Nrealizations) + '/'
    else:
        path = par.plots_dir + root_fig + par.root +\
            '_ts=' + str(par.time_step) +\
            '_L='+str(par.vector_length) + \
            '_'+ analy +'/mat'
    
    if 'tDeath' in to_modify:
        tdeath_seg = '_tthRan-' + \
            "{:.1f}".format(var.tDeaths[0])+'-' + \
            "{:.1f}".format(var.tDeaths[-1])
        name_ran = path + periode_seg + noiseLevel_seg + tdeath_seg
        return name_ran
        

    if not os.path.exists(path):
        os.makedirs(path)

    name_ran = path + periode_seg + noiseLevel_seg + tau_seg + pop_seg

    return name_ran 


def var_tagAndLabels(varName, values, var, par):

    labels = []

    if varName == 'periode':
        tag = '$T$[yr]'
        for e in values:
            labels.append("{:.0f}".format(e*par.time_step))

    elif varName == 'tau':
        tag = 'r$\tau$[yr]'
        # tag = '$\lambda$[yr$^-1$]'
        for e in values:
            labels.append("{:.2f}".format(par.time_step*e))

    elif varName == 'false_ratio':
        tag = '$F$[%]'
        for e in values:
            labels.append("{:.2}".format(e))

    elif varName == 'noiseLevel':
        tag = '$\sigma_{T}$[%]'
        for e in values:
            labels.append("{:.2}".format(e))

    elif varName == 'kError':
        tag = '$k_{\epsilon}$[%]'
        for e in values:
            labels.append("{:.2f}".format(e))
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, tau, false_ratio, kError')
        quit()

    return tag, labels


def set_title_mat(varNameX, varNameY, num, maxNum, var, par):

    if varNameX == 'periode' and varNameY == 'noiseLevel':
        if num == 0:
            return r'$\tau$ = ' + "{:4.0f}".format(var.tau*par.time_step)
        elif num == maxNum-1:
            return "{:4.0f}".format(var.tau*par.time_step) + '[yr]'
        else:
            return "{:4.0f}".format(var.tau*par.time_step)
    elif varNameX == 'periode' and varNameY == 'tau':
        return '$\sigma_{T}$ = ' + "{:2.1f}".format(100*var.noiseLevel) + '[%]'
    elif varNameX == 'noiseLevel' and varNameY == 'tau':
        if num == 0:
            return '$T$ = ' + "{:d}".format(var.periode)
        elif num == maxNum-1:
            return "{:d}".format(var.periode) + '[yr]'
        else:
            return "{:d}".format(var.periode)
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, tau, false_ratio, kError')
        quit()


