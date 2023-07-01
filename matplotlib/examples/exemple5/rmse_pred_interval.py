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
interval = []
lr       = []
arima    = []
lstm     = []

with open('result.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "intv":
            interval.append(int(row[0]))
            lr.append(float(row[1]))
            arima.append(float(row[2]))
            lstm.append(float(row[4]))
    csvfile.close()



############################################################################################

plt.title(r'Prediction accuracy', fontsize=14)
plt.plot(interval, lr, '.-', color='green', label='LR')
plt.plot(interval, arima, 'x-', color='goldenrod', label='ARIMA')
plt.plot(interval, lstm, '*-', color='red', label='LSTM')
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('Prediction interval (s)', fontsize=12)
plt.grid(True)
plt.axis([0, 75, 0, 2.5])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()