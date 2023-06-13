
import numpy as np


def print_halfLifes(Delta_t, observed_per_surb, name):

    #loss_percent = 2**(-Delta_t/(halfLife*par.time_step))*100
    halfLife_fun = -Delta_t/np.log2(observed_per_surb/100)
    meanLife_fun = -Delta_t/np.log(observed_per_surb/100)
    #print(' percentage lost in time', Delta_t,
    #    '[yr] given half life = ', halfLife*par.time_step, loss_percent)
    print(name, r'$t_{1/2}$ =',  halfLife_fun, ' given percent of  surb', observed_per_surb,  '% after ', Delta_t,
        '[yr]',)
    print(name, r'$\tau$ = ', meanLife_fun, ' given percent of  surb', observed_per_surb,  '% after ', Delta_t,
          '[yr] \n')

# ethnobotanical low
Delta_t = 9 # [yr]
observed_per_surb = 100-9 # 100 - 8
name = 'ethnobotanical low'
print_halfLifes(Delta_t, observed_per_surb, name)

# ethnobotanical high
Delta_t = 9  # [yr]
observed_per_surb = 100-26  # 100 - 8
name = 'ethnobotanical high'
print_halfLifes(Delta_t, observed_per_surb, name)

#military test
Delta_t = 0.115#[yr]
observed_per_surb = 100-17#100 - 8
name = 'military'
print_halfLifes(Delta_t, observed_per_surb, name)

# military perceptual
Delta_t = 1  # [yr]
observed_per_surb = 100-100*(80-52)/80  # 100 - 8
name = 'military perceptual'
print_halfLifes(Delta_t, observed_per_surb, name)

# military procedual-motor
Delta_t = 1  # [yr]
observed_per_surb = 100-100*(750-700)/750  # 100 - 8
name = 'military procedual-motor'
print_halfLifes(Delta_t, observed_per_surb, name)

# military salute low
Delta_t = 1  # [yr]
observed_per_surb = 100- 67  # 100 - 8
name = 'military salute low'
print_halfLifes(Delta_t, observed_per_surb, name)

# military don gas mask high
Delta_t = 1  # [yr]
observed_per_surb = 100-96  # 100 - 8
name = 'military don gas mask high'
print_halfLifes(Delta_t, observed_per_surb, name)


# CPR
Delta_t = 3  # [yr]
observed_per_surb = 12  # 100 - 8
name = 'CPR'
print_halfLifes(Delta_t, observed_per_surb, name)

#gaelic football
Delta_t = 6*7/365 # [yr]
observed_per_surb = 100 - 100*(15-12)/15  # 100 - 8
name = 'gaelic'
print_halfLifes(Delta_t, observed_per_surb, name)

# recall school
Delta_t = 26*7/365  # [yr]
observed_per_surb = 100 - 100*(80-58)/80  # 100 - 8
name = 'recall school'
print_halfLifes(Delta_t, observed_per_surb, name)

# recognition school
Delta_t = 26*7/365  # [yr]
observed_per_surb = 100 - 100*(85-80)/85  # 100 - 8
name = 'gaelic'
print_halfLifes(Delta_t, observed_per_surb, name)