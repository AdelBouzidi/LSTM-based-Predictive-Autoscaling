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
epoch = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
rmse   = [24.267, 22.825, 22.966, 19.186, 15.558, 26.630, 25.573, 15.781, 18.747, 16.280, 15.081, 14.738]





############################################################################################

plt.title(r'La precision de la prediction', fontsize=14)
plt.plot(epoch, rmse, '.-', color='green', label='RMSE')
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('Le nombre d"epoques d"entrainement', fontsize=12)
plt.grid(True)
plt.axis([0, 28, 10, 32])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()