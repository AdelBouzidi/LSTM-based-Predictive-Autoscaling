from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from keras.models import load_model
import time
import subprocess
import argparse
from influxdb import InfluxDBClient
import time, os
import datetime
from datetime import datetime, timedelta


'''Le script python: "autoscaler.py" prend en entrée: "monitored_cpu_usage", ensuite il prend les "timestep = 10" dernières
  valeurs de cpu_usage dans ce fichier pour construire le vecteur: input_x, ce vecteur sera autilisé comme entrée
  du Model LSTM: MODEL_LSTM(input_x) = yhat, yhat représente les "timestep = 10" valeurs de cpu_usage à prédire.
  Notes bien que, ce script utilise le modèle LSTM sauvegardé dans le fichier: "lstm_model.h5" et ne construit pas
  le modèle à chaque prédiction.'''

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

# la fonction insert_replicas permette d'insérer le nombre de replicas du sts: pas-ss

def insert_replicas(replicas, timestamp):
    #client.query('drop measurement replicas')
    json_body = [
        {
            "measurement": "replicas",
            "tags": {
                "host": "pas-ss-0"
            },
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "fields": {
                "replicas": replicas
            }
        }
    ]
    result = client.write_points(json_body)
    return result

# La fonction insert_input_x permette d'insérer le vecteur input_x dans le measurement: "cpu_usage_current"
# ce vecteur contient les 10 dernières utilisation du CPU, par exemple:

# 2022-07-14 20:00:00 ==> timestamp
# 2022-07-14 19:59:30
# 2022-07-14 19:59:00
# 2022-07-14 19:58:30
# 2022-07-14 19:58:00
# 2022-07-14 19:57:30
# 2022-07-14 19:57:00
# 2022-07-14 19:56:30
# 2022-07-14 19:56:00
# 2022-07-14 19:55:30

def insert_input_x(input_x, timestamp):
    for cpu_usage in input_x:
        json_body = [
            {
                "measurement": "cpu_usage_current",
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
        timestamp = timestamp - timedelta(hours=0, minutes=0, seconds=30)
    return result


# La fonction insert_yhat permette d'insérer le vecteur yhat dans le measurement: "cpu_usage_predicted"
# ce vecteur contient les 10 futures utilisation du CPU, par exemple:

# 2022-07-14 20:00:30 ==> timestamp + timedelta(hours=0, minutes=0, seconds=30)
# 2022-07-14 20:01:00
# 2022-07-14 20:01:30
# 2022-07-14 20:02:00
# 2022-07-14 20:02:30
# 2022-07-14 20:03:00
# 2022-07-14 20:03:30
# 2022-07-14 20:04:00
# 2022-07-14 20:04:30
# 2022-07-14 20:05:00

def insert_yhat(yhat, timestamp):
    for cpu_usage in yhat:
        timestamp = timestamp + timedelta(hours=0, minutes=0, seconds=30)
        json_body = [
            {
                "measurement": "cpu_usage_predicted",
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




# le modèle LSTM entrainé prend comme entrée un vecteur de taille: timestep, et donne comme sortie un vecteur
# de taille timestep.
# comme ce paramètre est utilisé dans plusieurs endroits dans le code, on l'a déclaré globalement ici.
timestep = 10



# cette fonction permet de récupérer un vecteur inpt de la forme: inpt.shape=(1,10,1), ensuite ce vecteur 
# sera utilisé comme entrée du modèle LSTM : LSTM(X)=Y

def get_input():
    query = 'select cpu_usage from monitored_cpu_usage order by time asc'
    result = client.query(query)
    data = []
    for point in result.get_points():
        data.append(int(point['cpu_usage']))
    return data[-timestep:]

# la fonction forecast permet de prédire les 10 prochaines valeurs dans le vecteur yhat, en donnant comme entrée 
# le vecteur input_x représentant les 10 dernières valeurs de cpu_usage
def forecast(model, inpt_list, n_input):
	data = array(inpt_list)
	data = data.reshape(timestep, 1)
	input_x = data[-n_input:, 0]
	input_x = input_x.reshape((1, len(input_x), 1))
	yhat = model.predict(input_x, verbose=0)
	yhat = yhat[0]
	return input_x, yhat

# cette fonction return True si une de 10 cpu_usage prédit dépasse un certain seuille: threshold
def is_scale_out(yhat, threshold):
    autoscale = False
    for i in range(len(yhat)):
        if yhat[i] > threshold:
            autoscale = True
    return autoscale

# la fonction get_replicas récupère le dernier nombre de replicas inséré dans le measurement "replicas" ==> order by time asc

def get_replicas():
    query = 'select replicas from replicas order by time asc'
    result = client.query(query)
    for point in result.get_points():
        state = point['replicas']
    return int(state)

# la fonction scale permette de mettre à l'echelle le sts "pas-ss" ==> si un seul POD, après exécution, ils seront 2 PODs, ...

def scale(replicas=1):
    proc1 = subprocess.Popen("kubectl scale sts pas-ss --replicas="+str(replicas)+"", shell=True, stdout=subprocess.PIPE)
    return proc1

# la fonction update_state permette de modifier la colonne scale du measurement scalestate à True ou False
# ensuite, le script "monitor_cpu.py" va consulter cette measurement pour savoir l'etat et orienter le stress

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

# la fonction get_ip_pod permette de récupérer l'adresse IP d'un certain POD "pod_name"

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

# la fonction init_all permette d'initialiser l'etat du scaling et l'ensemble de measurements: cpu_usage_current
# cpu_usage_predicted, replicas, le nombre de replicas. Elle est exécutée à chaque fois on lance autoscaler.py

def init_all():
    update_state(False)
    client.query('drop measurement cpu_usage_current')
    client.query('drop measurement cpu_usage_predicted')
    client.query('drop measurement replicas')
    timestamp = datetime.now()
    insert_replicas(1, timestamp)



print("connecting to influxd : ")


# connexion au serveur influxdb, ici on utiliser l'adresse IP du service, pas celle du POD, car l'adresse du POD 
# elle est dynamique et change à chaque démmarage du POD influxdb
client = connect("10.111.219.45", 8086)

# paramétrer le timezone à UTC
os.environ['TZ'] = 'UTC'
time.tzset()

# initialiser le système d'autoscaling
init_all()


# cpu_threshold c'est le seuil maximale d'utilisation du cpu, à partir de ce seuil, on peut considérer que
# le container est surchargé
cpu_threshold = 85

# nombre de replicas initialement est 1
replicas = 1
while True:
    # charger les 10 dernières consommations de cpu à partir du measurement: "monitored_cpu_usage"
    inpt_list = get_input()
    if len(inpt_list) >= timestep: 
        n_input = timestep * 1
        # on utilise seulement le modèle LSTM entrainé et sauvegardé dans le fichier: lstm_model.h5
        model = load_model('lstm_model.h5')
        input_x, yhat = forecast(model, inpt_list, n_input)
        
        print("===============>>>>>>>>>***********<<<<<<<<<<<<===============")
        print("yhat = LSTM_MODEL[input_x]")
        print("les ", timestep," dernières valeurs de cpu_usage input_x  : \n", input_x)
        print("les ", timestep," valurs de cpu_usage à prédir yhat : \n", yhat)
        
        input_x = input_x.reshape(10, 1)
        input_x = [x[0] for x in input_x]
        input_x = [input_x[len(input_x) - i] for i in range(1, len(input_x) + 1)]
        timestamp = datetime.now()
        insert_input_x(input_x, timestamp)
        insert_yhat(yhat, timestamp)
        replicas = get_replicas()
        if is_scale_out(yhat, cpu_threshold):
            # on doit faire ici l'action d'autoscaling, c'est à dire la création d'un nouveau replica
            # on a mis %5 pour forcer kubernetes à n'est pas dépasser 5 replicas, pour 
            # eviter de surcharger les machines virtuelles
            replicas = 1 + (replicas % 5)
            # modifier l'etat du scaling à True
            update_state(True)
            # insérer le nouveau nombre de replicas dans le measurement "replicas"
            insert_replicas(replicas, timestamp)
            # scaler le sts "pas-ss"
            proc1 = scale(replicas)
            time.sleep(2)
            proc1.kill()
            print("pas-ss has been scaled succesfuly !!!")
        else:
            insert_replicas(replicas, timestamp)
    else:
        print("La taille du vecteur: input_x is <  ", timestep, " !!!, yhat = LSTM_MODEL[input_x]")
    time.sleep(300)





