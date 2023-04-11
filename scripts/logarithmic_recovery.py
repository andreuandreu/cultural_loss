import matplotlib.pyplot as plt
import numpy as np


def k_tanh(t, k0, k1, d):
    return k0 + (k1 - k0)*np.tanh(t*d)

def k_sep(t, k0, k1, d):
    k_recovery =[k0]

    for i, e in enumerate(t):
        k_recovery.append(k_recovery[i] + np.log((k1 - k_recovery[i])*d))

    return k_recovery[:-1]


t = np.linspace(0, 1000, num=100)
k0 = 19
k1 = 22
d = 0.1

k_s = k_sep(t, k0, k1,d)
k_th = k_tanh(t, k0, k1, d)

plt.plot(t, k_th)
plt.plot(t, k_s)
plt.xlabel("t")
plt.ylabel("k(t)")
plt.show()
