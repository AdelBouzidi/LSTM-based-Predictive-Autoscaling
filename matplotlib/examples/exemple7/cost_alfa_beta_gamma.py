#!/usr/bin/env python3
from threading import Timer
from time import sleep
import os
import random
import schedule
import time
from influxdb import InfluxDBClient
from datetime import datetime
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import array


############################################## Cost #########################################
ep = []
cost30305 = []
cost60604 = []
cost1103 = []
with open('result.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            ep.append(float(row[0]))
            cost30305.append(float(row[1]))
            cost60604.append(float(row[2]))
            cost1103.append(float(row[3]))
    csvfile.close()

############################################################################################


plt.title(r'Mean Cost', fontsize=14)
plt.plot(ep, cost30305, '-', color='green', label=r'$\alpha = 0.5, \beta = 0.3, \gamma = 0.3$')
plt.plot(ep, cost60604, '-', color='red', label=r'$\alpha = 0.4, \beta = 0.6, \gamma = 0.6$')
plt.plot(ep, cost1103, '-', color='goldenrod', label=r'$\alpha = 0.3, \beta = 1, \gamma = 1$')
plt.ylabel('Mean Cost', fontsize=12)
plt.xlabel('Number of episodes', fontsize=12)
plt.grid(True)
plt.axis([-1, 122, 0, 30])
plt.legend(loc='best', ncol = 1, fontsize= '13')

plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.85, wspace=0.35, hspace=0.55)



plt.show()