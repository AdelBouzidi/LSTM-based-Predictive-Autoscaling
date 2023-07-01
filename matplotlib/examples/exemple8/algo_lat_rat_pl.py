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


plt.subplot(3, 1, 1)
plt.title(r'(a) Packet Loss', fontsize=14)
plt.bar(periodes, PL, align='center', width=0.4, color='red', alpha=0.5, label = 'Packet Loss')
plt.ylabel('Usage')
plt.tick_params(labelsize=11)
plt.ylabel('Packet Loss (%)', fontsize=11.5)
plt.grid(True)
plt.axis([0.5, 5.5, 0, 0.5])
plt.legend()
plt.legend(loc='best', fontsize= '12')
plt.xticks(periodes, ('HC', 'Reduced DTPRO', 'DTPROv1', 'DTPROv2', 'DTPRO'))
plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.92, wspace=0.35, hspace=0.55)



plt.subplot(3, 1, 2)
plt.title(r'(b) Latency', fontsize=14)
plt.bar(periodes, LATENCY, align='center', width=0.4, color='blue', alpha=0.5, label = 'Latency')
plt.ylabel('Usage')
plt.tick_params(labelsize=11)
plt.ylabel('Latency (ms)', fontsize=11.5)
plt.grid(True)
plt.axis([0.5, 5.5, 1.1, 1.8])
plt.legend()
plt.legend(loc='best', fontsize= '12')
plt.xticks(periodes, ('HC', 'Reduced DTPRO', 'DTPROv1', 'DTPROv2', 'DTPRO'))
plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.92, wspace=0.35, hspace=0.55)



plt.subplot(3, 1, 3)
plt.title(r'(c) Link Utilization', fontsize=14)
plt.bar(periodes, THROUGHPUT, align='center', width=0.4, color='green', alpha=0.5, label = 'Link Utilization')
plt.ylabel('Usage')
plt.tick_params(labelsize=11)
plt.ylabel('LU (MB/s)', fontsize=11.5)
plt.grid(True)
plt.axis([0.5, 5.5, 0, 10])
plt.legend()
plt.legend(loc='best', fontsize= '12')
plt.xticks(periodes, ('HC', 'Reduced DTPRO', 'DTPROv1', 'DTPROv2', 'DTPRO'))
plt.subplots_adjust(left=0.11, bottom=0.06, right=0.95, top=0.92, wspace=0.35, hspace=0.55)

plt.show()