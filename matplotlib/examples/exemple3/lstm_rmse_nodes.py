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
nodes = []
rmse   = []
loss   = []

with open('result1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "nodes":
            nodes.append(int(row[0]))
            rmse.append(float(row[1]))
            loss.append(float(row[2]))
    csvfile.close()



############################################################################################

plt.title(r'Prediction accuracy', fontsize=14)
plt.plot(nodes, rmse, '.-', color='green', label='RMSE')
plt.plot(nodes, loss, '.-', color='goldenrod', label='LOSS')
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('Number of hidden nodes', fontsize=12)
plt.grid(True)
plt.axis([0, 210, 0, 1.2])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)


plt.show()