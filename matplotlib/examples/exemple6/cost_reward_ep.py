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
cost24 = []
with open('result24.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            ep.append(float(row[0]))
            cost24.append(float(row[3]))
    csvfile.close()


cost60 = []
with open('result60.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            cost60.append(float(row[3]))
    csvfile.close()

cost120 = []
with open('result120.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            cost120.append(float(row[3]))
    csvfile.close()

############################################## Reward #########################################
reward24 = []
with open('result24.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            reward24.append(float(row[2]))
    csvfile.close()


reward60 = []
with open('result60.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            reward60.append(float(row[2]))
    csvfile.close()

reward120 = []
with open('result120.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "ep":
            reward120.append(float(row[2]))
    csvfile.close()

############################################################################################
reward24 = list(np.sort(array(reward24)))
reward60 = list(np.sort(array(reward60)))
reward120 = list(np.sort(array(reward120)))

# cost24 = list(np.sort(array(cost24)))
# cost60 = list(np.sort(array(cost60)))
# cost120 = list(np.sort(array(cost120)))


plt.subplot(2, 1, 1)
plt.title(r'(a) Mean Cost', fontsize=14)
plt.plot(ep, cost24, '-', color='green', label='act24')
plt.plot(ep, cost60, '-', color='red', label='act60')
plt.plot(ep, cost120, '-', color='goldenrod', label='act120')
plt.ylabel('Mean Cost', fontsize=12)
plt.xlabel('Number of episodes', fontsize=12)
plt.grid(True)
plt.axis([-1, 122, 0, 14])
plt.legend(loc='best', ncol = 3, fontsize= '13')

plt.subplots_adjust(left=0.12, bottom=0.08, right=0.95, top=0.92, wspace=0.35, hspace=0.55)

plt.subplot(2, 1, 2)
plt.title(r'(b) Mean Reward', fontsize=14)
plt.plot(ep, reward24, '-', color='goldenrod', label='act24')
plt.plot(ep, reward60, '-', color='red', label='act60')
plt.plot(ep, reward120, '-', label='act120')
plt.ylabel('Mean Reward', fontsize=12)
plt.xlabel('Number of episodes', fontsize=12)
plt.grid(True)
plt.axis([-1, 122, 0, 6])
plt.legend(loc='best', ncol = 3, fontsize= '13')

plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.92, wspace=0.35, hspace=0.55)





plt.show()