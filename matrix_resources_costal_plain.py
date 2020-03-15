# -*- coding: utf-8 -*-
"""model of resource allocation in desertic costal plain"""
from __future__ import division
import time
import datetime
import sys
import colorsys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from random import randint
from collections import deque
from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import matplotlib
from subprocess import call
import matplotlib.ticker as ticker



import pickle
import cmath

#def make_grid(n,m):


def land_grid(m,l):
    '''create grid'''
    
    land_matrix = np.full((m, l), cnt.max_land)#np.random.rand(6).reshape(m,l)*cnt.land_0
    
    print('t_o land', land_matrix)
    return land_matrix


class constants:

    land_0 = 2.5
    land_productivity = 0.2
    max_land = 5
    sea_productivity = 0.9
    consumption_rate = 1
    L_threshold = 0.4
    time = 10
    length = 3
    width = 2


def time_steps():
    
    t = np.arange(1,cnt.time, 0.5)
    
    return t

def jump(L):

    '''jumping strategy, returns the cell with the most resources'''

    aux = np.where(L == max(L))

    return aux[0][0], aux[0][1]

def area_jump(L, m, l, d):
    '''jumping strategy, within an area defined by a distance d, returns the cell with the most resources'''

    low_lim = m-d
    if low_lim <  0:
        low_lim = 0

    high_lim = m+d
    if high_lim >  cnt.length-1:
        high_lim = cnt.length-1
        
  
    area = L[low_lim:high_lim][:]
    aux = np.where(area == max(area))



    return aux[0][0], aux[0][1]





def resorurces_evol(t, c, L, m=0, l=0):

    '''operates the matrix of resources'''
    resources = []
    sea_consumption = []
    production = []
    position = []
    resources.append(L)
    position.append([m,l])
    
    for e in t:

        p = cnt.land_productivity* (1 - L/cnt.max_land) *L
        L = L + p

        if L[m][l] - c > cnt.L_threshold:
           
            L[m][l] =  L[m][l] - c
            s = 0
            print('land consumption')        
        
        else:  
            if m == 0 and L[m][l] - cnt.L_threshold + cnt.sea_productivity > c+0.2:       
                
                land_margin = L[m][l] - cnt.L_threshold
                L[m][l] = cnt.L_threshold 

                s = c  - land_margin
                print('sea/land consumption')
                
                
            else:
                m = randint(0,cnt.width-1)
                l = randint(0,cnt.length-1)
                s = -1
                print ('jump to another square', m, l)


        resources.append(L)
        position.append([m,l])
        sea_consumption.append(s)
        production.append(p)
    for e in resources:
        print('eee', e, '\n', '\n')
    return resources, sea_consumption, production, position
    

def plot_resources(resources, name):

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.set_xlabel("t")
    ax.set_ylabel("Sea resources used")

    ax.plot(resources)
    name_sea = '../plots_costal_resources/sea_resources_'+ name + '.eps'
    plt.savefig(name_sea,  bbox_inches = 'tight')

def plot_matrix(matrix, name):

    
    fig = plt.figure(name)
    ax = fig.gca() #fig.add_subplot(111)
    M = ax.matshow(matrix.T, interpolation='nearest', cmap=cm.OrRd)#,
    
    plt.colorbar(M, ax=ax)
    
    

def wtf(M):

    resources = []
    resources.append(M)
    
    
    for i in range(3):

        p = 2*M
        M = M +p #np.random.rand(6).reshape(2,3)
        
        print('rmatrix', M )
        print('resuurces', resources)
        resources.append(M)

 
    return resources

def matrix_movie(M, position, nom):
    fig, ax = plt.subplots()#111, 'matrix movie'
    A = M[0].T
    ax.clear()
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=cnt.max_land)
    matrice = ax.matshow(A, cmap = cm.OrRd, norm = normalize)
    ax.scatter(position[0][0], position[0][1], marker = 'o', facecolors = 'k')#.plot(position[0])
    plt.colorbar(matrice)
    #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    
    def update(i):
        #matrice.axes.clear()
        A = M[i].T
        #matrice.axes.clear()

        if i>0:
            #color = A[position[i-1][1], position[i-1][0]]
            #print( color, 'aaa', position[i-1], A )
            matrice.axes.scatter(position[i-1][0], position[i-1][1], marker = 'o', c = 'w', lw = 0,  s=33)#
        matrice.set_array(A)
        print (i, 'iiii', len(position), len(M))
        matrice.axes.scatter(position[i][0], position[i][1], marker = 'o', facecolors = 'k', lw = 0, s=30)
        

    ani = animation.FuncAnimation(fig, update, frames=len(M), interval=630)#

    
    plt.show()
    name_gif = 'matrix_land_' + nom+'.gif'
    ani.save(name_gif,  dpi = 80)#,writer = 'imagemagick')


    #ax01.set_title('$\\nu$ ' + str(v) + ' b ' + str(b))
    # set label names
    #ax.set_xlabel("m")
    #ax.set_ylabel("l")
    


cnt = constants()
t= time_steps()
Land = land_grid(cnt.width, cnt.length)
name = sys.argv[1]
#wtf(Land)
#example_matrxani()
resources, sea_consumption, production, position = resorurces_evol(t, cnt.consumption_rate, Land , 0, 0)
plot_resources(sea_consumption, 'sea_consumption')
#plot_matrix(resources[0], 'land_resources')

matrix_movie(resources, position, name)

plt.show()













