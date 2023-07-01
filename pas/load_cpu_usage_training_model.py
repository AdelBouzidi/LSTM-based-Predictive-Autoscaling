import argparse
from influxdb import InfluxDBClient
import time
import datetime
from datetime import datetime, timedelta
import csv
from pandas import read_csv
from numpy import array
from datetime import datetime

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

# la fonction insert_point permette d'insérer un seul enregistrement dans un measurement

def insert_point(timestamp, cpu_usage, measurement):
    json_body = [
        {
            "measurement": measurement,
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
    return result

# la fonction get_ip_pod permette de recupérer l'adresse IP d'un POD: "pod_name"

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

# connecter au serveur influxdb
print("connecting to influxd : ")
client = connect("10.111.219.45", 8086)

# supprimer l'ancien model enregistré, pour eviter de duppliquer les données
client.query('drop measurement cpu_usage_training_model')


# lire le modèle initialement enregistré dans le fichier CSV: "cpu_usage_training_model.csv", dans dataset
dataset = read_csv('cpu_usage_training_model.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])


data = dataset.values
data = array(data)

# le modèle contient 1400 valeurs, donc elle correspondent à 11 heures et 40, sachant que l'interval entre
# chaque 2 cpu_usage est 30s
# donc la première valeure corresponde à la date actuelle - 11 heures et 40 minutes
timestamp = datetime.now() - timedelta(hours=11, minutes=40, seconds=0)

# insertion des données dans influxdb
for index in range(len(data)):
    insert_point(timestamp, data[index], "cpu_usage_training_model")
    timestamp = timestamp + timedelta(hours=0, minutes=0, seconds=30)

print("Le Modele d'entrainement est bien chargé sur InfluxDB")

