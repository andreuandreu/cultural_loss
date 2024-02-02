import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.stats
from scipy.stats import binom
from numba import jit, float64, types, int64, bool_
from scipy import stats
#from sklearn.preprocessing import normalize



def setup(N, Ps):
    #cost
    Clower, Cupper, Cmean, Cscale, Cnum = 0, 10, 0, 3, N
    Cdist = stats.truncnorm(a= (Clower-Cmean)/Cscale, b=(Cupper-Cmean)/Cscale, loc=Cmean, scale=Cscale).rvs(Cnum)

    #Effectiveness
    Elower, Eupper, Emean, Escale, Enum = 0, 3, 0.1,  0.1, N
    Edist = stats.truncnorm(a= (Elower-Emean)/Cscale, b=(Eupper-Emean)/Escale, loc=Emean, scale=Escale).rvs(Enum)
    
    #Impact
    #Ilower, Iupper, Iscale, Inum = 100, 100, 1, N
    #Idist = stats.truncexpon(b=(Iupper-Ilower)/Iscale, loc=Ilower, scale=Iscale).rvs(Inum)

    Ilower, Iupper, Imean, Iscale, Inum = 45, 55, 50,  3, N
    Idist = stats.truncnorm(a= (Ilower-Imean)/Iscale, b=(Iupper-Imean)/Iscale, loc=Imean, scale=Iscale).rvs(Inum)

    #probability
    Plower, Pupper, Pscale, Pnum = 0, 3, 0.1, Ps
    Pdist = stats.truncexpon(b=(Pupper-Plower)/Pscale, loc=Plower, scale=Pscale).rvs(Pnum)

    

    fig, ax = plt.subplots(1, 1)
    ax.hist( Pdist, density=True, bins='auto', histtype='stepfilled', alpha=0.2)

    return Cdist, Idist, Edist, Pdist


def update_CandE(tradeoffs, Cdist, Edist, N):

    threshold =  np.max(tradeoffs) + 0.15 * np.max(tradeoffs)
    thresholdPr = 10
    auxTh = np.where( tradeoffs > threshold)
    Escale = 0.01
    Cscale = 0.01


    #cost
    numSel = int(N/thresholdPr)
    ind = np.argpartition(tradeoffs, -numSel )[-numSel :]
    newCmean = np.sum((1/tradeoffs)**2*Cdist)/(N*np.mean((1/tradeoffs)**2))
    newCmean2 = np.mean(Cdist[ind])
    
    #print('new mean', newCmean, newCmean2 ,'old mean',  np.mean(Cdist), np.mean(tradeoffs))
    aux = np.where( tradeoffs == np.max(tradeoffs) )
    newCscale2 =  np.std(Cdist)
    #print ('max trad', threshold , 'old scale', 3, 'new scale2', newCscale, newCscale2  )

    tradeoffWeights = 1/abs(tradeoffs)
    norm = tradeoffWeights/sum(tradeoffWeights)
    #Cdist= np.random.choice(Cdist, len(Cdist), p= tradeoffWeights)
    newCmeanWeighted = np.random.choice(Cdist, 1, p= norm ) + np.random.normal(loc= 0, scale=Cscale, size=len(Cdist))
    
    

    Clower, Cupper, Cmean, Cscale, Cnum = 0, newCmeanWeighted+30, newCmeanWeighted, newCscale2, N
    Cdist = stats.truncnorm(a= (Clower-Cmean)/Cscale, b=(Cupper-Cmean)/Cscale, loc=Cmean, scale=Cscale).rvs(Cnum)

    #Effectiveness
    newEmean = np.sum((1/tradeoffs)**2*Edist)/(N*np.mean((1/tradeoffs)**2))
    newEmean2 =np.mean(Edist[ind])#np.mean(Edist)# 

    #print('new mean', newEmean, newEmean2, 'old mean',  np.mean(Edist), np.mean(tradeoffs))
    #newEscale = np.std(Edist[auxTh[0]])
    #newEscale = np.std((1/tradeoffs)**2*Edist/ (1/tradeoffs)**2)
    newEscale2 =  0.1#p.std(Edist)#[ind]
    #print('max trad', threshold , 'old scale', 0.1, 'new scale', newEscale, newEscale2 )   

    #weightedEs = Edist/np.log(abs(tradeoffs))
    #norm = weightedEs/sum(weightedEs)
    #print('sususus', sum(norm))
    Edist= np.random.choice(Edist, len(Edist), p= norm) + np.random.normal(loc= 0, scale=Escale, size=len(Edist))

   
    #Elower, Eupper, Emean, Escale, Enum = 0, 3, newEmean2,  newEscale2, N
    #Edist = stats.truncnorm(a= (Elower-Emean)/Escale, b=(Eupper-Emean)/Escale, loc=Emean, scale=Escale).rvs(Enum)

    #Impact
    Imean = np.random.rand()*100
    Ilower, Iupper, Iscale, Inum = Imean-Imean*0.1, Imean+Imean*0.1, 3, N
    Idist = stats.truncnorm(a= (Ilower-Imean)/Iscale, b=(Iupper-Imean)/Iscale, loc=Imean, scale=Iscale).rvs(Inum)

    return Cdist, Edist, Idist

    

def plot_payoff(preparing_cost, Payoffs, labels, cOpt, payoffOpt, Effectiv, event_prob, impact, tag):

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig, ax = plt.subplots(1, 1)

    for p, l in zip(Payoffs, labels):
        ax.plot(preparing_cost, p, label = tag + "{:.2g}".format(l))
        aux = np.where(p == max(p))
        ax.scatter(preparing_cost[aux[0]], max(p))
        if 'I' in tag:
            tagI = 'Impact for $E$ = ' + "{:.2f}".format(Effectiv) + ' $P_e$ = ' + "{:.2f}".format(event_prob) 
            ax.set_title(tagI)
        if 'E' in tag:
            tagE = 'Effectiveness for $P_e$ = ' + "{:.2f}".format(event_prob) + ' $I$ = '  +  "{:.0f}".format(impact)  
            ax.set_title(tagE)
        if 'P' in tag:
            tagP = 'Probability for $E$ = ' + "{:.2f}".format(Effectiv) + ' $I$ = '  +  "{:.0f}".format(impact) 
            ax.set_title(tagP)
       
    ax.plot( cOpt,  payoffOpt, lw = 4, ls = '--', c = '0.5')
    ax.set_ylabel('Payoff')
    ax.set_xlabel("Cost")

    ax.legend()

def plot_scatters(Cdist, Idist, Edist, PayoffDist, Ne):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(Cdist, PayoffDist, alpha = 0.5, c= 'y')
    ax1.set_ylabel('Payoff')
    ax1.set_xlabel("Cost")

    ax2.scatter(Idist, PayoffDist, alpha = 0.5, c = 'r')
    ax2.set_xlabel("individual Impact")

    ax3.scatter(Edist, PayoffDist, alpha = 0.5, c = 'g')
    ax3.set_xlabel("Effectivenes")

    ax3.set_title("event n: " + str(Ne))


def plot_means_std(i, fig, ax1, ax2, meanCs, meanEs, stdCs, stdEs):
    
    alpha = 0.2
    
    if i == 1:
        ax1.plot(meanCs,label = 'Costs', alpha = alpha, c = 'y')
        ax1.plot(meanEs,label = 'Effectiv', alpha = alpha, c = 'g')
    else:
        ax1.plot(meanCs, alpha = alpha, c = 'y')
        ax1.plot(meanEs, alpha = alpha, c = 'g')
    
    
    ax1.set_ylabel("means")
    ax1.set_xlabel("event number")

    ax2.plot(stdCs, alpha = alpha, c = 'y')
    ax2.plot(stdEs, alpha = alpha, c = 'g')
    ax2.set_ylabel("standard deviations")
    ax2.set_xlabel("event number")

    #fig.suptitle(r'floating $\sigma_E$')

    fig.suptitle(r'$\sigma_E = 0.1$')
    ax1.legend(frameon=False)
    #ax2.legend(frameon=False)

def plot_wealth(time, wealth):

    fig, ax1 = plt.subplots(1,1)
    ax1.plot(time, wealth)
 
    ax1.set_ylabel("means")
    ax1.set_xlabel("event number")


def plot_cOpt(cOpt_p,cOpt_i, cOpt_e, p, e, i, event_prob, impact, Effectiv):

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig, ax = plt.subplots(1, 1)

    tagE = ': $P_e$ = ' + "{:.2f}".format(event_prob) + ' $I$ = '  +  "{:.0f}".format(impact) 
    tagP = ': $E$ = ' + "{:.2f}".format(Effectiv) + ' $I$ = '  +  "{:.0f}".format(impact) 
    tagI = ': $E$ = ' + "{:.2f}".format(Effectiv) + ' $P_e$ = ' + "{:.2f}".format(event_prob)

    ax.plot( p, cOpt_p, label = 'Prob' + tagP)
    ax.plot( e, cOpt_e, label = 'Effecti' + tagE)
    ax.plot( i, cOpt_i, label = 'Impact/100' + tagI)

    aux = np.where(cOpt_i > 0)[0][0]
    ax.scatter(i[aux-1], 0)
    aux = np.where(cOpt_e > 0)[0][0]
    ax.scatter(e[aux-1], 0)
    aux = np.where(cOpt_p > 0)[0][0]
    ax.scatter(p[aux-1], 0)

    aux = np.where(cOpt_e == max(cOpt_e))[0][0]
    ax.scatter(e[aux], cOpt_e[aux])


    cOpt_max = np.max(np.concatenate([cOpt_e, cOpt_i, cOpt_p]).ravel()) 

    ax.set_ylim(-0.1, cOpt_max+0.1)

    ax.set_ylabel('Optimal Cost')
    #ax.set_xlabel('Effectivenes/Probability/Impact $\cdot$ 0.01')
    

    ax.legend()

def c_optimal( event_prob, impact, Effectiv):
    cOpy  = np.log(impact*event_prob*Effectiv)/Effectiv
    return cOpy

def c_optimal4( event_prob, impact, Effectiv):
    cOpy  = np.log(impact*event_prob*Effectiv)*impact/Effectiv
    return cOpy
    
def payoff_units(preparing_cost, event_prob, impact, Effectiv):
    Payoff = - (preparing_cost/event_prob+impact*np.exp(-Effectiv*preparing_cost))
    return Payoff


def preparedness(Effectiv, preparing_cost):
    Preparedness = 1-np.exp(-Effectiv*preparing_cost)
    return Preparedness


def wealth(wMax, w0, growth, time):

    ws = []
    w = w0
    for t in time:
        if w >= wMax:
            w = wMax 
        else: 
            w += (1-np.exp(-t*growth))
        ws = np.append(ws, w)
    return ws

def cost_function():

    M = 100
    time = np.linspace(0, 1000)
    

    fig, ax = plt.subplots(1, 1)
    #ax.plot(time, cost1, c = 'b')
    for m in range(0, M, 10):
        cost2 = (np.exp(-time/m))#/(time)
        ax.plot(time, cost2)


###things to implement

'''
- cost function, either exponential decay or hipervolic tangent to decay to default setup after a Memory threshold is corssed
-- if cost function implemented, then measure total cost of costs up to memory threshold. i.e. any invesment will have a max cost remembered
- wealth function to recover
- selection of new costs/effectiveness, improve the method
- scenarios with longer/shorter memories
- memories are dependent on size of the system and determine probabilityies to compute tradeoffs/payoffs
- selection on effectiveness?
'''


def main():

    #C - Investment in preparedness (cost) [eco/yr]
    #p – Probability of climatic event [eve/yr]
    #I – Climatic event impact [eco/eve]
    #E - Effectivenes of cost, how much prepared yu are per unit of cost [yr/eco]
    #W - wealth recovery [eco/yr]
    #wMax - max wealth [eco]
    #M - memory span, 1/M is the min probability that can be remembered

    preparing_cost = np.arange(0, 0.07,0.0002)*100
    Effectivs =  np.arange(0.01,2,0.2)
    impacts =  np.arange(0.1,2,0.2)*100
    event_probs = np.logspace(-1.6, 0.1, 10)

    #cost_function()

    Payoffs_effe = np.empty((len(Effectivs),len(preparing_cost))) * np.nan
    Payoffs_prob = np.empty((len(Effectivs),len(preparing_cost))) * np.nan
    Payoffs_impact = np.empty((len(Effectivs),len(preparing_cost))) * np.nan
    
    event_prob = 0.1 #[event/yr]
    impact = 20 #[% of GDP, population]
    Effectiv = 0.8
    Nagents = 500
    Nevents = 444
    nTrials = 11
    NProbabilities = Nevents
    time = np.arange(1000)
    
    wealthGrowth = 0.3
    wMax = 100
    memory = 100 

    Cdist0, Idist0, Edist0, Pdist0 =  setup(Nagents, NProbabilities)
    #Pdist0[0] = 1

    prob0 = Pdist0[int(np.random.rand(1)*len(Pdist0))]
    if prob0 < 1/memory:
        prob0 = 1/memory

    print('prob', prob0, 'is this min prob', 1/memory )
    
    PayoffDist = payoff_units(Cdist0, prob0, Idist0, Edist0)
    plot_scatters(Cdist0, Idist0, Edist0, PayoffDist, prob0)
    PayoffDist = payoff_units(Cdist0, 0.1, Idist0, Edist0)
    plot_scatters(Cdist0, Idist0, Edist0, PayoffDist, 0.1)
    PayoffDist = payoff_units(Cdist0, 0.3, Idist0, Edist0)
    plot_scatters(Cdist0, Idist0, Edist0, PayoffDist, 0.3)

    

    fig, (axMeans, axSd) = plt.subplots(1,2)
    for i in range(nTrials):
        Cdist, Edist = Cdist0, Edist0
        meanCs, meanEs = [],[]
        stdCs, stdEs = [], []
        for i, p in enumerate(Pdist0):
            Cdist, Edist, Idist = update_CandE(PayoffDist, Cdist, Edist, Nagents)
            meanCs = np.append(meanCs, np.mean(Cdist))
            meanEs = np.append(meanEs, np.mean(Edist))
            stdCs = np.append(stdCs, np.std(Cdist))
            stdEs = np.append(stdEs, np.std(Edist))

            p_e = np.mean(Pdist0[:i])
            PayoffDist = payoff_units(Cdist, p, Idist0, Edist)#p = 0.5
            
            #if i<4:#i%66 == 1:
            #    plot_scatters(Cdist, Idist0, Edist, PayoffDist, i)

            #if i>len(Pdist0)-5:#i%66 == 1:
            #    plot_scatters(Cdist, Idist0, Edist, PayoffDist, i)
        
        
        plot_means_std(i, fig, axMeans, axSd, meanCs, meanEs, stdCs, stdEs)
    ws = wealth(wMax, 1, 0.1, time)
    #plot_wealth(time, ws)



    #for j, (e, im, p) in enumerate(zip(Effectiv, impact, event_prob)):

    #    Payoffs_n_effect[j] =  payoff(preparing_cost, event_prob, impact, e)
    cOpt_p = np.empty(len(event_probs))
    cOpt_i = np.empty(len(impacts))
    cOpt_e = np.empty(len(Effectivs))
    payoff_optimal_p = np.empty(len(event_probs))
    payoff_optimal_i = np.empty(len(impacts))
    payoff_optimal_e = np.empty(len(Effectivs))


    for j, E in enumerate(Effectivs):
        Payoffs_effe[j] =  payoff_units(preparing_cost, event_prob, impact, E)
        c_op = c_optimal( event_prob, impact, E)
        if c_op < 0:
            cOpt_e[j] = 0
        else:
            cOpt_e[j] = c_op
        payoff_optimal_e[j] =  payoff_units(cOpt_e[j], event_prob, impact, E)


    for j, P in enumerate(event_probs):
        Payoffs_prob[j] =  payoff_units(preparing_cost, P, impact, Effectiv)
        c_op =  c_optimal( P, impact, Effectiv)
        if c_op < 0:
            cOpt_p[j] = 0
        else:
            cOpt_p[j] = c_op
        payoff_optimal_p[j] =  payoff_units(cOpt_p[j], P, impact, Effectiv)

    for j, I in enumerate(impacts):
        Payoffs_impact[j] =  payoff_units(preparing_cost, event_prob, I, Effectiv)
        c_op = c_optimal(event_prob, I, Effectiv)
        if c_op < 0:
            cOpt_i[j] = 0
        else:
            cOpt_i[j] = c_op
        payoff_optimal_i[j] =  payoff_units(cOpt_i[j], event_prob, I, Effectiv)

    
    #ws = wealth(wMax, 1, t)
        
    #plot_payoff(preparing_cost, Payoffs_effe, Effectivs, cOpt_e, payoff_optimal_e,  Effectiv, event_prob, impact, tag = '$E$ ='  )
    #plot_payoff(preparing_cost, Payoffs_prob, event_probs, cOpt_p,  payoff_optimal_p, Effectiv, event_prob, impact, tag = '$P_e$ =')
    #plot_payoff(preparing_cost, Payoffs_impact, impacts, cOpt_i, payoff_optimal_i,  Effectiv, event_prob, impact, tag = '$I$ =' )

    #plot_cOpt(cOpt_p, cOpt_i, cOpt_e, event_probs, Effectivs, impacts/100, event_prob, impact, Effectiv)
    plt.show()

if __name__ == '__main__':
    main()