# Standalone simple linear regression example
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from numpy import array



tb0 = [4.57577, 4.21744, 4.87816, 4.86681, 8.34925, 2.419, -0.09583, 3.0635, 3.17228, 4.87049, -8.401, -4.59069,
       43.22714, 48.51041, 29.44356, 33.80421, 33.36405, 39.97507, 31.37091, 9.33456, 22.36946, 19.01101, 17.67275]
tb1 = [0.00115, 0.0408, 0.04315, 0.05477, -0.00233, 0.09437, 0.13286, 0.09905, 0.10022, 0.07925, 0.19114, 0.1597,
       -0.15969, -0.19117, -0.07927, -0.10023, -0.09909, -0.1329, -0.09443, 0.00236, -0.0548, -0.04313, -0.04079]

xtb0 = pd.Series(tb0)
xtb1 = pd.Series(tb1)

int = np.arange(len(xtb0))


plt.subplot(1, 2, 1)
plt.title(r'(a) $\lambda$ parameter', fontsize=14)
plt.xlabel('Data interval number')
plt.ylabel('$\lambda$')
plt.plot(int, xtb0, '^-', color='goldenrod', label='$\lambda$')
#ax1.plot(int, xtb0, '^-', color='goldenrod', label='B0 Parameter')
plt.grid(True)
plt.axis([-1, 23, -15, 60])
plt.legend(loc='best', ncol = 1, fontsize= '13')
plt.subplots_adjust(left=0.1, bottom=0.20, right=0.95, top=0.88, wspace=0.35, hspace=0.55)

plt.subplot(1, 2, 2)
plt.title(r'(b) $\mu$ Parameter', fontsize=14)
plt.xlabel('Data interval number')
plt.ylabel(r'$\mu$')
plt.plot(int, xtb1, 'o-', color='green', label=r'$\mu$')
#ax2.plot(int, xtb1, 'o-', color='green', label='B1 Parameter')
plt.grid(True)
plt.axis([-1, 23, -0.4, 0.4])
plt.legend(loc='best', ncol = 1, fontsize= '13')
plt.subplots_adjust(left=0.12, bottom=0.18, right=0.95, top=0.90, wspace=0.35, hspace=0.55)

plt.show()