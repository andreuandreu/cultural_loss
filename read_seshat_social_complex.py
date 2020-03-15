#code to read seshat cvs file containing the rows and columns for 8 social complex caracteristics

#NGA,PolID,Time,PropCoded,Pop,Terr,Cap,Hier,Gov,Infra,Info,Money,Dupl,Uniq

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import sys
import matplotlib.ticker as ticker
import pandas as pd

 

def getColumns(inFile, delim="\t", header=True):
    """
    Get columns of data from inFile. The order of the rows is respected
    
    :param inFile: column file separated by delim
    :param header: if True the first line will be considered a header line
    :returns: a tuple of 2 dicts (cols, indexToName). cols dict has keys that 
    are headings in the inFile, and values are a list of all the entries in that
    column. indexToName dict maps column index to names that are used as keys in 
    the cols dict. The names are the same as the headings used in inFile. If
    header is False, then column indices (starting from 0) are used for the 
    heading names (i.e. the keys in the cols dict)
    """
    cols = {}
    indexToName = {}
    for lineNum, line in enumerate(inFile):
        if lineNum == 0:
            headings = line.split(delim)
            i = 0
            for heading in headings:
                heading = heading.strip()
                if header:
                    cols[heading] = []
                    indexToName[i] = heading
                else:
                    # in this case the heading is actually just a cell
                    cols[i] = [heading]
                    indexToName[i] = i
                i += 1
        else:
            cells = line.split(delim)
            i = 0
            for cell in cells:
                cell = cell.strip()
                cols[indexToName[i]] += [cell]
                i += 1
                
    return cols, indexToName


def polis_selection(cols, name):
    '''
    given a pols name e.g. Larium, selects all the rows that have that police as first element
    input: complete datafile, name of polis
    output: a trimed out dataset only cointaining the characteristics of that one polis
    '''

    #ind = np.where(cols[indexToName[0]]== name)
    #aux_rows = cols.T
    #polis_rows = aux_rows[ind]
    polis_columns = cols[indexToName[name]] 

    return polis_columns

def polis_names(df):
    '''
    returns an array of the names of the existant polices as defined by the first column with the tag NGA
    '''
    names = [df['NGA'][0]]
    i = 0
    for e in df['NGA']:
        if e != names[i]:
            names.append(e)
            i += 1
            #print(names, i, 'name number', i)
    return names
        



dataframe = pd.read_csv(sys.argv[1], delimiter = ',')
names = polis_names(dataframe)

df_polis = dataframe[dataframe.NGA == names[0]]
#print(df_polis)
df_polis.sample(10) #print sample
df_polis.head()


def plot_all_parameters(time, df_polis):
    '''
    for a given polis dataset it plots the values of all the relevant parameters against the temporal column
    '''
    fig = plt.figure()
    plt.rcParams.update({'font.size': 15})    
    ax = fig.add_subplot(111)#

    ax.set_xlabel("whatever")
    ax.set_ylabel("t")
    for e in df_polis:
        ax.plot(time, e)
df_polis[['PropCoded','Pop']]

plot_all_parameters(df_polis['Time'], df_polis[['PropCoded','Pop','Terr','Cap','Hier','Gov','Infra','Info','Money']])
plt.show()
   