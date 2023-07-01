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

############################################################################################
timesteps = ["1","2","3","4","5","6","7","8","9","10"]
rmse = [12.2, 10.0, 7.3, 8.4, 8.5, 10.4, 13.7, 15.2, 17.1, 17.6]

x = np.arange(len(rmse))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, rmse, width, label='RMSE', color='deeppink')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE')
ax.set_xlabel('La liste des Timesteps')
ax.set_title('RMSE par timesteps')
ax.set_xticks(x)
ax.set_xticklabels(timesteps)
plt.axis([-1, 10, 5, 20])
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

