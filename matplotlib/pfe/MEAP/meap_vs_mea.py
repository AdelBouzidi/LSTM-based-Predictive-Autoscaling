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
for i in range(1,101):
    timesteps.append(i)

cpu_with_prediction = [13, 13, 15, 18, 18, 20, 8, 9, 13, 15,
                       21, 23, 26, 29, 28, 28, 28, 20, 20, 21,
                       25, 29, 32, 35, 38, 39, 40, 31, 35, 35,
                       38, 43, 43, 46, 49, 47, 51, 38, 43, 44,
                       48, 53, 52, 54, 56, 56, 61, 60, 50, 50,
                       54, 55, 58, 60, 65, 67, 69, 69, 59, 63,
                       64, 62, 68, 69, 74, 75, 79, 76, 71, 70,
                       70, 72, 72, 74, 74, 76, 30, 35, 36, 39,
                       43, 42, 45, 51, 49, 50, 39, 44, 45, 49,
                       52, 51, 55, 56, 61, 61, 61, 49, 48, 54]
cpu_without_prediction = [13, 13, 15, 18, 18, 20, 10, 10, 13, 15,
                          21, 24, 26, 30, 27, 30, 20, 21, 20, 25,
                          25, 29, 32, 36, 38, 39, 40, 30, 33, 35,
                          37, 42, 42, 46, 50, 49, 50, 38, 44, 45,
                          45, 48, 52, 52, 55, 56, 62, 60, 49, 50,
                          54, 55, 58, 59, 70, 67, 67, 67, 61, 62,
                          64, 63, 67, 69, 74, 76, 78, 78, 70, 69,
                          69, 73, 71, 72, 73, 73, 76, 75, 74, 68,
                          82, 83, 83, 85, 85, 83, 89, 89, 91, 88,
                          88, 92, 91, 91, 93, 95, 93, 91, 92, 92]
############################################################################################

plt.title(r'La precision de la prediction', fontsize=14)
plt.plot(timesteps, cpu_without_prediction, '.-', color='gold', label='CPU Sans Prédiction')
plt.plot(timesteps, cpu_with_prediction, '.-', color='green', label='CPU Avec Prédiction')

#plt.scatter(learning_rate, rmse, c='b')

plt.ylabel('Consommation de CPU', fontsize=12)
plt.xlabel('Timesteps', fontsize=12)
plt.grid(True)
plt.axis([-2, 103, 0, 115])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()

