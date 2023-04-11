We want to model cultural "memory" decay in the context of a trait not being practiced for a certain amount of time. This will be the functioning of a trait inside a node in a cultural network.

The assumption is that, as long as the trait is performed, it does not decay.
Then once the trait is not performed (be it for lack of resources, dependence on other cultural traits, out of fashion, etc.)

The trait starts to decay in a given way. Our initial assumption is a log-normal way depending on a decay rate $∂$ and an initial knowledge $k_0$

Then, once the trait is performed again (be it because the resources are back, the fashion comes back, the dependence is fulfilled...) its knowledge/memory is replenished to initial values. 

This model has the following parameters:

- ∂ decay rate
- $f$ frequency of performance (for example, access to resources in a seasonal basis)
- $n$ noise level, i.e. the departure from a non-perfectly regular frequency 
- $k_0$ initial knowledge $= 1$
- $k_{th}$ knowledge threshold $= 0.1 \cdot k_0$
- $t_s$ time step $= 0.1$ 

## Cultural decay function
insptred from:
 - Halbwachs, M. On Collective Memory (Univ. Chicago Press, Chicago, 1992)
 - Assmann, J. in Cultural Memory Studies. An International and Interdisciplinary Handbook (eds Erll, A. & Nünning, A.) 109–118 (Walter de Gruyter, Berlin, 2008).
 -  https://doi.org/10.1038/s41562-018-0474-5 The universal decay of collective memory and attention, Canadia 2019

[![Pasted image 20230406104757.png](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406104757.png)](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406104757.png)

[![Pasted image 20230406104709.png](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406104709.png)](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406104709.png)

The cultural memory only part  on their model (neglecting the communicative memory) can be summarised as a log-norm decay


$\frac{dk}{dt} = - k\cdot ∂$,
$k(t)=k_0e^{-t∂} = e^{c-t∂},$

where, $d$ on the left side, first expression, is the derivative of knowledge over time.  $∂$ is the decay rate, $k_0$ accounts for the initial knowledge in our rendering of the model. 


## Embers (or quest for fire) model: trait decay and recovery 

If $k(t) < k_{th} \enspace \rightarrow k(t)=0$, where $k_{th}$ is a knowledge threshold beyond which, if the knowledge is not enough, the trait is considered to be lost. *For the Olympiads is a complex one, as it was alive in living memory, even though the event was not performed for thousands of years, but it needed other prerequisites before recreating the games, like Greek independence and the revival of athletics.* Thus, in theory,  $k_{th}$ can be arbitrarily low.


$$
\begin{equation}
k(t) =  
	\begin{cases}
		e^{c-∂ \Delta t}  + [k_0 - k(t-t_s)] \cdot E(t) \enspace \text{if} \enspace k(t-t_s) > k_{th}  \\
		0 \enspace \text{if} \enspace k(t-t_s) < k_{th} ,
	\end{cases}
\end{equation}
$$

where 

$$
\begin{equation}
E(t) = (t+n(t))  \bmod f
\end{equation}
$$


thus, $E(t)$ can have the values

$$
\begin{equation}
E(t) = 
	\begin{cases}
		1 \\
		0 
	\end{cases}
\end{equation}
$$

and $n(t)$ is a noise value normally distributed around $\mu = 0$ with variance $\sigma =t_s$ $n(t)=N(0, t_s^2) \enspace \exists \enspace [-\infty, \infty]$. 
$c = \log(k_0)$. 
$\Delta t$ is the time interval since the last time E(t) = 1, i.e. how many time steps $t_s$ ago the last expression event happened. For example, the Olympiads happen every 4 years, thus $\Delta t = f = 4$, but in Covid year they were delayed one year, so $\Delta t = 5$. In the case of WWI $\Delta t =8$ and for WWII $\Delta t =12$.


For practical proposes, I'm considering that the threshold is one order of magnitude lower than the initial condition, or $k_{th} = 1/10$.

Then, for a continously recurring event: $E(t) = 1$ &rarr; $\Delta t_{max} = f_{th} =-\log(k_{th}/k_0)/∂=-[\log(k_{th})-c]/∂$,
$f_{th}(k_{th}=0.1\cdot k_0) = -2.3/∂$


### One realization for $f=5, t_s = 0.01, ∂=2.3/f, n = 0.66$

[![Pasted image 20230406100048.png](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406100048.png)](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406100048.png)

[![Pasted image 20230406095941.png](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406095941.png)](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406095941.png)

### For 33 realizations

[![Pasted image 20230406101123.png](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406101123.png)](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406101123.png)


### For 3333 realizations 

[![Pasted image 20230406210711.png](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406210711.png)](https://github.com/andreuandreu/cultural_loss/blob/master/notes/Pasted%20image%2020230406210711.png)

# After-loss Innovation model

This model expands on the embers model with the assumption that a cultural trait can be reinvented and culturally acquired after it has disappeared from the cultural memory.

It assumes that there is an Innovation rate $I(t)$ where there is a non-zero chance of recreating the initial cultural trait that was lost from cultural memory but can be found in the archaeological record. An example is the discontinuity of the use of concrete for about 1500 years in Western Europe [HISTORY OF CONCRETE FROM ROMAN TIMES TO THE EIGHTEENTH CENTURY, JANET IRENE ATKINSON, 1979]. Though, in this case, the occurrence of natural events where the resources might be available can be considered continuous, what is broken is the trade network to link resources and knowledge.  

This model has the following parameters:
- $A_{th}$ acquisition threshold
- $a$ acquisition rate of a lost trait
- $\sigma$ variance on acquisition 

The model can be descrived by:

$$
\begin{equation}
k(t) = 
	\begin{cases}
		e^{c-\Delta t ∂} \enspace \text{if} \enspace k(t)  < k_{th} \\
		I(t) \enspace \text{if} \enspace k(t)  > k_{th}
	\end{cases}
\end{equation}
$$

where $I(t) = k_0 \cdot E(t)\cdot A(t)$

$$
\begin{equation}
A(t) = 
	\begin{cases}
		1 \enspace \text{if} \enspace N(a, \enspace \sigma)  < A_{th} \\
		0 \enspace \text{if} \enspace N(a, \enspace \sigma)  > A_{th}
	\end{cases}
\end{equation}
$$

$A(t)$ is the acceptation or not of a newly innovated previously lost cultural trait. 
Thus, given the conditions are right, i.e. $E(t) = 0$
It would be accepted with a probability depending on whether the normal distribution $N(a,\enspace\sigma)$ is bigger or smaller than a set threshold of acquisition $A_{th}$. 
$a$ is the mean acquisition rate
$\sigma$ is the variance on the acquisition 



