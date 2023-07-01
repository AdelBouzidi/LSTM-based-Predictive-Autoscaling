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
taille_vecteur = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
rmse   = [16.546, 23.139, 19.675, 19.679, 21.394, 17.212, 15.902, 19.059, 17.123, 20.760, 13.793, 16.098, 16.203, 14.063, 13.571, 17.659, 12.545, 15.593, 14.522]

############################################################################################

plt.title(r'La precision de la prediction', fontsize=14)
plt.plot(taille_vecteur, rmse, '.-', color='red', label='RMSE')
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('La taille du vecteur H', fontsize=12)
plt.grid(True)
plt.axis([0, 220, 10, 25])
plt.legend(loc='best', ncol = 3, fontsize= '13')
plt.subplots_adjust(left=0.11, bottom=0.16, right=0.95, top=0.89, wspace=0.35, hspace=0.55)

plt.show()