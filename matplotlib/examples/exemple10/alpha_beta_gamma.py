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
pdq  = [r'$Latency \times 8$', r'$Load \times 3$', r'$ICF$']
rmse = []
forcast = [0.96,1.05,0.94,1.14,1.13,1.29,0.88,0.92,1.06,1.01]
size_200_1_1 = [0.379 * 8, 5.803 * 3, 31.865] # latency
size_1_200_1 = [1.64  * 8, 1.884 * 3, 31.865] # load
size_1_1_200 = [0.949 * 8, 8.51  * 3, 21.865] # ICF

i = 0

# with open('result.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         if row[0] != "pdq" and i % 10 == 0 and i < 100:
#             pdq.append(str(row[0]))
#             rmse.append(round(float(row[1]),2))
#         i = i + 1
#     csvfile.close()



############################################################################################
x = np.arange(len(size_200_1_1))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, size_200_1_1, width, label=r'$\alpha = 200, \beta = 10,   \gamma = 10$'  )
rects2 = ax.bar(x ,        size_1_200_1, width, label=r'$\alpha = 10,   \beta = 200, \gamma = 10$'  )
rects3 = ax.bar(x + width, size_1_1_200, width, label=r'$\alpha = 80,   \beta = 40,   \gamma = 200$')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performances')
#ax.set_title('Clustering and Controllers placement performances')
ax.set_xticks(x)
ax.set_xticklabels(pdq)
plt.axis([-1, 3, 0.6, 40])
plt.grid(True)
plt.xticks(rotation=0)

#plt.xlabel('ARIMA p,d,q values', fontsize=10, labelpad=1)
ax.legend(loc=2)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.92, wspace=0.35, hspace=0.55)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in range(len(rects)):
        hgt = rects[rect].get_height()
        if rect == 0:
            height = round(hgt / 8, 2)
        if rect == 1:
            height = round(hgt / 3, 2)
        if rect == 2:
            height = round(hgt, 1)
        print (height)
        ax.annotate('{}'.format(height),
                    xy=(rects[rect].get_x() + rects[rect].get_width() / 2, hgt),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

#fig.tight_layout()

plt.show()

