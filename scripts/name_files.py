import os


def file_name_n_varValue(to_modify, value, var, par, analy = ''):

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

    elif to_modify == 'pop':
        var.pop = value
        
    else:
        print('wrong to modify name argumetn!!! be carefull, only options are periode, halfLife, false_ratio, kError')
        quit()
   

    path = par.output_dir + par.root + '_ts='+str(par.time_step) + '_L=' + str(par.vector_length) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

  

    name = path + analy +\
    'T='   + "{:.2f}".format(var.periode) + \
    '_Te=' + "{:.2f}".format(var.noiseLevel) + \
    '_t12='+ "{:.1f}".format(var.halfLife)+ \
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

    if 'halfLife' in periode_seg:
        halfLife_seg = '_t12ran-' + \
            "{:.1f}".format(var.halfLifes[0]) + \
            '-'+"{:.1f}".format(var.halfLifes[-1])
    else:
        halfLife_seg = '_t12-' + "{:.1f}".format(var.halfLife)

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
        noiseLevel_seg + halfLife_seg   # + kError_seg  +
    print('nonanonaonaoa', name_fig)

    return name_fig



def name_survival_ranges(to_modify, root_fig, var, par, analy = ''):
    
    if 'periode' in to_modify:
        periode_seg = '_Tran-'+"{:.2f}".format(var.periodes[0])+'-'+"{:.2f}".format(var.periodes[-1])
    else:
        periode_seg = '_T-' + "{:.2f}".format(var.periode)

    if 'halfLife' in periode_seg:
        halfLife_seg = '_t12ran-'+"{:.1f}".format(var.halfLifes[0])+'-'+"{:.1f}".format(var.halfLifes[-1])
    else:
        halfLife_seg = '_t12-' + "{:.1f}".format(var.halfLife)

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

    if not os.path.exists(path):
        os.makedirs(path)

    name_ran = path + periode_seg + noiseLevel_seg + halfLife_seg + pop_seg

    return name_ran 


def var_tagAndLabels(varName, values, var, par):

    labels = []

    if varName == 'periode':
        tag = '$T$[yr]'
        for e in values:
            labels.append("{:.0f}".format(e*par.time_step))

    elif varName == 'halfLife':
        tag = 'r$\tau$[yr]'
        # tag = '$\lambda$[yr$^-1$]'
        for e in values:
            labels.append("{:.2f}".format(par.time_step*e))

    elif varName == 'false_ratio':
        tag = '$T_f$[%]'
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


def set_title_mat(varNameX, varNameY, num, maxNum, var, par):

    if varNameX == 'periode' and varNameY == 'noiseLevel':
        if num == 0:
            return r'$\tau$ = ' + "{:4.0f}".format(var.halfLife*par.time_step)
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


