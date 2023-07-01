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


############################################## RB #########################################
learning_rate = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
rmse   = [24.819, 23.245, 27.004, 23.774, 23.653, 35.638, 24.955, 26.982, 24.678, 21.053]
timesteps = []
for i in range(1,6):
    timesteps.append(i)

cpu = [76, 76, 76, 78, 85, 89]
threshold = [85, 86, 87, 88, 89, 90]
############################################################################################

plt.title(r'La precision de la prediction', fontsize=14)
plt.plot(threshold, cpu, '.-', color='red', label='MAX CPU Par Threshold')

#plt.scatter(learning_rate, rmse, c='b')

plt.ylabel('Consommation de CPU', fontsize=12)
plt.xlabel('Threshold', fontsize=12)
plt.grid(True)
plt.axis([84.5, 90.5, 75, 92])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()

