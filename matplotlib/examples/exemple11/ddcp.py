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

periodes = [1,2,3,4,5]

PL = [0.420, 0.261, 0.167, 0.145, 0.114]
LATENCY = [1.634, 1.328, 1.291, 1.257, 1.204]
THROUGHPUT = [9.03, 7.698, 7.073, 6.597, 5.976]

CD_kmeans = [1.629, 5.471, 0.578]
CD_dqn = [1.816, 0.691, 0.379]

CL_kmeans = [8.618, 6.714, 5.971]
CL_dqn = [8.385, 6.924, 5.803]

ICF_kmeans = [35.517, 35.129, 31.865]
ICF_dqn = [36.949, 34.955, 31.001]

labels = ["p = 4", "p = 5", "p = 6"]
x = np.arange(len(CD_kmeans))  # the label locations
width = 0.23  # the width of the bars

# fig, ax = plt.subplots()
# ax.plot(range(1, 14), wcss, color='green', label=r'WCSS')
# ax.plot(range(4, 7), wcss[3:6], 'o', color='red', label='Optimum number \n of clusters')
# ax.set_title('WCSS of KMeans models under \n different number of clusters ', fontsize=14)
# ax.set_ylabel('WCSS')
# ax.set_xlabel('Number of clusters')
# ax.grid(True)
# ax.legend(loc='best', ncol=1, fontsize='13')
# plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.85, wspace=0.35, hspace=0.55)
# plt.show()



fig, axs = plt.subplots(3, 1)

print (axs.shape)

rects_CD_kmeans = axs[0].bar(x - width/2, CD_kmeans, width, label='CD Reduced DDCP')
rects_CD_dqn = axs[0].bar(x + width/2, CD_dqn, width, label='CD DDCP', color='green')

axs[0].set_ylabel('Control Delay (ms)')
axs[0].set_title('Mean Control Delay (CD)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].axis([-0.5, 2.5, 0.0, 6])
axs[0].grid(True)
#plt.xlabel('ARIMA p,d,q values', fontsize=10, labelpad=1)
axs[0].legend(loc='best', ncol=1, fontsize='13')


rects_CL_kmeans = axs[1].bar(x - width/2, CL_kmeans, width, label='CL Reduced DDCP')
rects_CL_dqn = axs[1].bar(x + width/2, CL_dqn, width, label='CL DDCP', color='green')

axs[1].set_ylabel('Control Load')
axs[1].set_title('Mean Control Load (CL)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].axis([-0.5, 2.5, 4, 10])
axs[1].grid(True)
#plt.xlabel('ARIMA p,d,q values', fontsize=10, labelpad=1)
axs[1].legend(loc='best', ncol=1, fontsize='13')
plt.subplots_adjust(left=0.11, bottom=0.06, right=0.95, top=0.7, wspace=0.35, hspace=0.55)


rects_ICF_kmeans = axs[2].bar(x - width/2, ICF_kmeans, width, label='ICF Reduced DDCP')
rects_ICF_dqn = axs[2].bar(x + width/2, ICF_dqn, width, label='ICF DDCP', color='green')

axs[2].set_ylabel('Intra-Cluster Factor')
axs[2].set_title('Mean Intra-Cluster Factor (ICF)')
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)
axs[2].axis([-0.5, 2.5, 29, 41])
axs[2].grid(True)
#plt.xlabel('ARIMA p,d,q values', fontsize=10, labelpad=1)
axs[2].legend(loc='best', ncol=1, fontsize='13')




plt.subplots_adjust(left=0.11, bottom=0.06, right=0.95, top=0.93, wspace=0.35, hspace=0.45)

plt.show()