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
batch_size = [8, 16, 32, 64, 128, 256, 384, 512, 768, 894, 1024]
rmse   = [19.634, 14.522, 22.342, 15.753, 18.710, 22.874, 16.693, 22.411, 20.792, 19.449, 19.543]

############################################################################################

plt.title(r'La precision de la prediction', fontsize=14)
plt.plot(batch_size, rmse, '.-', color='gold', label='RMSE')
plt.scatter(batch_size, rmse, c='b')
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('La taille du batch d"entrainement', fontsize=12)
plt.grid(True)
plt.axis([0, 1050, 12, 26])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()

