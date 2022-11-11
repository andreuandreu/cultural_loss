import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm




def generate_gaussian(mu, sigma, xa, xb):
    # calculate the z-transform
    za = ( xa - mu ) / sigma
    zb = ( xb - mu ) / sigma

    x = np.arange(za, zb, 0.01) # range of x in spec
    x_all = np.arange(-10, 10, 0.01) # entire range of x, both in and out of spec
    # mean = 0, stddev = 1, since Z-transform was calculated
    y = norm.pdf(x,0,1)
    y2 = norm.pdf(x_all, mu, sigma)

    return x_all, y2


# define constants
mu1 = 0.0
sigma1 = 1.0
xa1 = 800
xb1 = 1200
a1 = 1#1.3

x_vec1, gauss1 = generate_gaussian(mu1, sigma1, xa1, xb1)

# define constants
mu2 = -0.5
sigma2 = 0.21
xa2 = xa1
xb2 = xb1
a2 =0.1

x_vec2, gauss2 = generate_gaussian(mu2, sigma2, xa2, xb2)

fig, ax = plt.subplots(figsize=(9,6))
plt.style.use('fivethirtyeight')
ax.plot( a1*gauss1)
ax.plot( a2*gauss2)

ax.plot(a1*gauss1 - a2*gauss2)


#ax.fill_between(x,y,0, alpha=0.3, color='b')
#ax.fill_between(x_all,y2,0, alpha=0.1)
ax.set_xlim([500,1500])
#ax.set_xlabel('# of Standard Deviations Outside the Mean')
#ax.set_yticklabels([])
#ax.set_title('Normal Gaussian Curve')

plt.savefig('gaussians_rested.png', dpi=72, bbox_inches='tight')
plt.show()