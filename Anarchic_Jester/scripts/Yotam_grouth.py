import random
import numpy
import matplotlib.pyplot as plt

def sim(np = 100000, nu = 900000, fps = 0.95, fus = 1, fpe = 0.95, fue = 0.75, k = 1000000, r = 1.01, mu = 0.01,
        t_max = 5000, freq_e = 0.05):

    #np: number of prepared individuals
    #nu: number of prepared individuals
    #fps: fitness of prepared individuals under stability
    #fus: fitness of unprepared individuals under stability
    #fpe: fitness of prepared individuals under climatic events
    #fue: fitness of unprepared individuals under climatic events
    #k: carrying capacity (when competition takes place)
    #r: reproductive rate
    #t_max: simulation length
    #freq_e: frequency of environmental events  

    nps = [np]  #we collect the numbers of prepared and unprepared
                #individuals throughout the simulation
    nus = [nu]
    
    for t in range(t_max): #simulation runs for t_max time units
        if random.random() < freq_e: #climatic event occured
            np = numpy.random.binomial(n=np, p = fpe)   #the new numbers are drawn from
                                                        #a binomial distribution based on fitness
            nu = numpy.random.binomial(n=nu, p = fue)
        else: #normal year
            max_n = (nu+np)*r
            if max_n < k: #when population size is below k, no competition (for simplicity)
                np = np*r
                nu = nu*r
            else: #when the population reaches carrying capacity competition takes place
                prob_p = np*fps/(np*fps+nu*fus) #the probability of an offspring to be prepared
                                                #is calculated based on the frequencies of the
                                                #two types and their fitness values
                np = numpy.random.binomial(n=k, p = prob_p)
                nu = k - np
        nps += [np]
        nus += [nu]
    x=range(t_max+1)
    plt.stackplot(x,nps,nus,labels = ["prepared", "Unprepared"])
    plt.legend(loc='upper left')
    plt.show()
    return None

sim()