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
timesteps = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
rmse   = [13.288, 14.552 ,12.837, 11.829, 6.461, 7.471, 4.855, 11.969, 15.197, 11.936, 13.107, 11.519, 12.376, 11.896, 10.751, 11.273, 14.290, 19.975]




############################################################################################

plt.title(r'La precision de la prediction', fontsize=14)
plt.plot(timesteps, rmse, '.-', color='darkorange', label='RMSE')
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('la taille de la fenêtre de prédiction', fontsize=12)
plt.grid(True)
plt.axis([0, 23, 2, 22])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()