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
import random
import matplotlib


############################################## RB #########################################
pdq  = []
rmse = []
forcast = [0.96,1.05,0.94,1.14,1.13,1.29,0.88,0.92,1.06,1.01]
i = 0

with open('result.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] != "pdq" and i % 10 == 0 and i < 100:
            pdq.append(str(row[0]))
            rmse.append(round(float(row[1]),2))
        i = i + 1
    csvfile.close()



############################################################################################
pred_met = ["LSTM","ARIMA","LR"]
nbr_congest = [157, 102, 83]

x = np.arange(len(nbr_congest))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, nbr_congest, width, label='Number predicted congestion')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('NPC')
ax.set_title('Number predicted congestion')
ax.set_xticks(x)
ax.set_xticklabels(pred_met)
plt.axis([-0.5, 2.5, 0, 200])
plt.grid(True)
plt.xticks(rotation=0)

#plt.xlabel('ARIMA p,d,q values', fontsize=10, labelpad=1)
ax.legend(loc=1)

plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.88, wspace=0.35, hspace=0.55)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
# autolabel(rects2)

#fig.tight_layout()

plt.show()

