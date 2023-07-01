import csv
from pandas import read_csv
import time
from numpy import array
import datetime
from datetime import datetime, timedelta
import subprocess
from cpu_load_generator import load_single_core, load_all_cores, from_profile
import psutil
import os
import random
import argparse
from influxdb import InfluxDBClient

# La fonction connect permette de connecter avec le serveur influxdb

def connect(host='localhost', port=8086):
    """Instantiate a connection to the InfluxDB."""
    user = 'admin'
    password = 'admin'
    dbname = 'pas'
    dbuser = 'admin'
    dbuser_password = 'admin'
    client = InfluxDBClient(host, port, user, password, dbname)
    return client

# la fonction get_training_data permette de récupérer le modèle d'entrainement depuis le measurement:
# "cpu_usage_training_model"

def get_training_data():
    query = 'select cpu_usage from cpu_usage_training_model order by time asc'
    result = client.query(query)
    data = []
    for point in result.get_points():
        data.append(point['cpu_usage'])
    return data

# la fonction get_state permette de récupérer l'etat du scaling (True/False)

def get_state():
    query = 'select scale from scalestate'
    result = client.query(query)
    for point in result.get_points():
        state = point['scale']
    return eval(str(state))

# la fonction update_state permette de  modifier la colonne scale du measurement scalestate à True ou False
# pour orienter le stress

def update_state(scale=False):
    client.query('drop measurement scalestate')
    timestamp = datetime.now()
    json_body = [
        {
            "measurement": "scalestate",
            "tags": {
                "host": "pas-ss-0"
            },
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "fields": {
                "scale": scale
            }
        }
    ]
    result = client.write_points(json_body)
    return result

# la fonction insert_cpu_usage permette d'insérer un seul enregistrement dans le measurement "monitored_cpu_usage"
# comme, on fait le stress, ensuite le monitoring chaque 30s, à chaque valeur monitorée, on appelle cette fonction

def insert_cpu_usage(cpu_usage, timestamp):
    json_body = [
        {
            "measurement": "monitored_cpu_usage",
            "tags": {
                "host": "pas-ss-0"
            },
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "fields": {
                "cpu_usage": cpu_usage
            }
        }
    ]
    result = client.write_points(json_body)

# la fonction clear_cpu_usage permette de suprimer le measurement monitored_cpu_usage, elle est utilisée pour
# faire l'initialisation et eviter de merger les donées des autres exécutions précedentes ...

def clear_cpu_usage():
    result = client.query('drop measurement monitored_cpu_usage')
    return result
# la fonction adjust_measurement permette de corriger la valeur de CPU usage monitorée

def adjust_measurement(list_cpu_usage, model_value):
    dist = 10000 - model_value
    adjust_cpu_usage = model_value
    for cpu_usage in list_cpu_usage:
        if abs(cpu_usage - model_value) < dist:
            dist = abs(cpu_usage - model_value)
            adjust_cpu_usage = cpu_usage
    return adjust_cpu_usage

# la fonction get_ip_pod permette de récupérer l'adresse IP d'un POD: "pod_name"

def get_ip_pod(pod_name):
    proc_pod_ip = subprocess.Popen("kubectl get pod "+pod_name+" --template '{{.status.podIP}}'", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,  preexec_fn=os.setsid)
    pod_ip_line = proc_pod_ip.communicate()[0]
    pod_ip_line = str(pod_ip_line)
    pod_ip = ""
    index = 2
    while index < len(pod_ip_line) - 1:
        pod_ip += pod_ip_line[index]
        index += 1
    return pod_ip

# la fonction stress_cpu_monitor permette d'exercer un stress sur le POD "pas-ss-0" d'un pourcentage: "percentage" et
# pendant une durée: "duration"
# ensuite, elle fait le monitoring, en utilisant "Kubernetes Metrics Server" via la commande: kubectl top pod | grep pas-ss-0

def stress_cpu_monitor(percentage = 50, duration = 10):
    try:
        # > /dev/null 2 > /dev/null
        proc_stress = subprocess.Popen("kubectl exec  --namespace=default pas-ss-0 -- bash -c 'stress-ng -c 1 -l "+str(percentage)+" -t "+str(duration)+"s'", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,  preexec_fn=os.setsid)
        cpu_usage_list = []
        for i in range(10):
            proc_monitor = subprocess.Popen("kubectl top pod | grep pas-ss-0", shell=True, stdout=subprocess.PIPE)
            cpu_usage_line = proc_monitor.communicate()[0]
            cpu_usage_line = str(cpu_usage_line)
            index = 10
            cpu_usage = ''
            while cpu_usage_line[index] != 'm':
                cpu_usage += cpu_usage_line[index]
                index += 1
            cpu_usage_list.append(float(cpu_usage))
            time.sleep(3)
            proc_monitor.kill()
        proc_stress.kill()
        cpu_usage_core = int(1000 * (percentage / 100))
        return adjust_measurement(cpu_usage_list, cpu_usage_core)
        #return round(max(cpu_usage_list), 2)
    except:
        return 0


# init_all initialise le nombre de replicas, l'etat du scaling à "False", et supprimer le measurement: "monitored_cpu_usage"

def init_all():
    proc1 = subprocess.Popen("kubectl scale sts pas-ss --replicas=1", shell=True, stdout=subprocess.PIPE)
    update_state(False)
    clear_cpu_usage()
    return proc1
	

# data contient le modèle de données pour stresser le CPU

client = connect("10.111.219.45", 8086)


print("Please wait while initializing the cluster !!!")
# appel à init_all(), ensuite le kill c'est pour arriter le processus généré par subprocess
proc1 = init_all()
data = get_training_data()
time.sleep(30)
proc1.kill()

print("Cluster initialized succefully !!!")


#i = random.randint(0, int(len(data)/100) - 1) * 100 + 30
i = 60
while True:
    state = get_state() 
    if state :
        i = random.randint(0, int(len(data)/1000) - 1) * 100 + 30
        update_state(False)
    
    print("la valeur de i est : ", i)
    print("CPU will be loaded by : ", data[i], " %")
    
    # on va charger le cpu par chaque valeur récupérée depuis influxdb: int(data[i]), 30) 
    cpu_usage = str(int(((stress_cpu_monitor(int(data[i]), 30))*100)/1000))
    
    print("cpu_usage : ", cpu_usage, " %")
 
    # insérer la valeur de CPU usage monitorée dans le measurement: "monitored_cpu_usage" dans "influxdb"
    timestamp = datetime.now()
    insert_cpu_usage(cpu_usage, timestamp)
    i = (i + 1) % len(data)


