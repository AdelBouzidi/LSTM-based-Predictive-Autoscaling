from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras.models import load_model
import csv
import time
import datetime
from datetime import datetime, timedelta
import subprocess
import os
import random
import argparse
from influxdb import InfluxDBClient
from tensorflow import keras

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
    data = array(data)
    data = data.reshape(len(data), 1)
    return data

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

# Le script python: "many_to_many.py" consiste à construire le modèle LSTM, en testant plusieurs paramétres, et en
# faisant la comparaison, ces paramètres seront utilisés dans le script: "build_model.py". Ce script prend en
# entrée les données d'entrainement depuis influxdb. Notes bien que ce script est exécuté manuellement pour
# but de test et comparaison.


# le modèle LSTM entrainé prend comme entrée un vecteur de taille: timestep, et donne comme sortie un vecteur
# de taille timestep.
# comme ce paramètre est utilisé dans plusieurs endroits dans le code, on l'a déclaré globalement ici.
timestep = 10

# la fonction split_dataset prend comme entreé tous les données de l'entrainement récupérées depuis influxdb
# "data". la forme "shape" de data est data.shape=(1400, 1) 1400 lignes et 1 colonne (cpu_usage). 
# Ensuite, cette fonction répartis 'split' l'ensemble de 1400 lignes en 2: train et test, tel que: 
# 1300 lignes pour train: train.shape=(1300,1), et 100 lignes pour test: test.shape(100,1). 
# Ensuite, elle regroupe les 1300 lignes de train sur 130 vecteurs, chaque vecteur de taille "timestep=10"
# en utilisant train = array(split(train, len(train)/timestep)).
# Donc la forme 'shape' de train devient train.shape=(130,10,1). La meme chose pour test: test.shape=(10.10.1) 

# en générale, si le nombre de lignes dans data divisible par timestep=10, on peut directement regrouper par timestep
# sinon on élimine certain valeurs en utilisant le modulo "%", par exemple nombre de lignes 1405, up_to_ts = 5
# train contient les données (1..1300), test contient (1300..1400) et les 5 seront éliminés 

#def split_dataset(dataset):
#    nbr_global_timestep = int(len(dataset)/timestep)
#    test_size = 10 * timestep
#    train_size = nbr_global_timestep * timestep - test_size 
#    data = dataset[:-(len(dataset) % timestep)]
#    train, test = data[:-test_size], data[-test_size:]
#    train = array(split(train, len(train)/timestep))
#    test = array(split(test, len(test)/timestep))
#    return train, test

def split_dataset(dataset):
    nbr_global_timestep = int(len(dataset)/timestep)
    test_size = 10 * timestep
    train_size = nbr_global_timestep * timestep - test_size
    z=len(dataset) % timestep
    if z != 0:
        data = dataset[:-(len(dataset) % timestep)]
    else:
        data = dataset
    train, test = data[:-test_size], data[-test_size:]
    train = array(split(train, len(train)/timestep))
    test = array(split(test, len(test)/timestep))
    return train, test


# Notes bien que le RMSE représente la distance entre la valeur prédite at la valeur actuelle

# Cette fonction evaluate_forecasts prend comme entree 2 vecteurs, le premier vecteur: actuel, qui est le vecteur
# test returné par la fonction split_dataset, sa forme est: actual.sahpe=(10,10,1). Le deuxième vecteur est
# predicted qui a aussi le meme shape predicted.shape=(10,10,1).
# Ensuite dans l'instruction: "for i in range(actual.shape[1]):" il va mésurer le Root Mean Square Error (RMSE)
# pour chaque vecteur parmi les 10 vecteurs, chaque valeur de RMSE est sauvegardée dans la liste: scores 
# (scores.append(rmse)), cette dernière liste va contenir 10 valeurs: scores=[rmse1, rmse2, rmse3, rmse4, rmse5, 
# rmse6, rmse7, rmse8, rmse9, rmse10]
# Le RMSE est calculé comme suite: on actual=[actual1, actual2, ..., actual10]
# predicted=[predicted1, predicted2, ..., predicted10], ensuite rmse1 = RMSE((actual1[0],predicted1[0]), 
# (actual2[0],predicted2[0]), ..., (actual10[0],predicted10[0])),  rmse2 = RMSE((actual1[1],predicted1[1]), 
# (actual2[1],predicted2[1]), ..., (actual10[1],predicted10[1])) ... jusqu'à rmse10

# la deuxième boucle permet de calculer le globale RMSE, la déffirence avec le premier RMSE c'est que le 
# premier calcule le RMSE pour chaque periode, c'est à dire on a 10 vecteurs, on essaye de calculer le RMSE 
# pour la première valeur dans tous les vecteurs, ensuite la deuxième, ..., jusqu'à la 10eme valeurs
# le deuxème RMSE consiste à mésurer une seule valeur qui la distance entre deux vecteurs de taille 100
# dont chaque vecteur est la concaténation des 10 sous vecteurs.

def evaluate_forecasts(actual, predicted):
    try:
        scores = list()
        # calculer le RMSE pour chaque période parmi les 10 périodes et sur les 10 vecteurs dans actual et predicted
        for i in range(actual.shape[1]):
                # on utilise la bibliothèque déjà importée pour calculer le MSE
                mse = mean_squared_error(actual[:, i], predicted[:, i])
                # le RMSE c'est la racine de MSE
                rmse = sqrt(mse)
                # sauvegarder chaque valeur de RMSE dans la liste : scores
                scores.append(rmse)
        # calculer le global RMSE
        s = 0
        for row in range(actual.shape[0]):
                for col in range(actual.shape[1]):
                        s += (actual[row, col] - predicted[row, col])**2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores
    except:
        for i in range(actual.shape[1]):
            scores.append(0.0)
        s = 0
        for row in range(actual.shape[0]):
                for col in range(actual.shape[1]):
                        s += (actual[row, col] - predicted[row, col])**2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores

# cette fonction permet d'afficher le RMSE globale et pour chaque période
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# comme la taille de train est 130, elle n'est pas suffisante pour faire l'antrainement, la fonction split_dataset
# permet de répartir les données de cette façon: [1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], ...
# de cette manière on peut pas avoir beaucoup de données pour l'antrainement, la solution est de décaler par 
# juste 1 step et ça donne:[1,2,3,4,5,6,7,8,9,10], [2,3,4,5,6,7,8,9,10,11], [3,4,5,6,7,8,9,10,11,12], ...

# cette fonction concerne seulement train
# Ensuite cette fonction sauvegarde chaque vecteur [1,2,3,4,5,6,7,8,9,10] dans X, et le vecteur qui vient après
# [11,12,13,14,15,16,17,18,19,20] dans y

def to_supervised(train, n_input, n_out=timestep):
	# la forme actuelle de train est train.shape=(130, 10, 1), l'instruction suivante permet de flatten
        # data pour avoir une nouvelle forme train.shape=(1300,1)
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# on met pour chaque itération de la boucle for un vecteur de taille 10 dans X
        # le vecteur qui vient après dans y, et on décale d'une seule position: in_start += 1
	for _ in range(len(data)):
		in_end = in_start + n_input
		out_end = in_end + n_out
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		in_start += 1
	return array(X), array(y)

# la fonction build_model permet de construire le modèle LSTM en précisant plusieurs parametres, elle return le modèle
# après entrainement dans: "return model", aussi elle permet de sauvegarder le modèle entrainé dans un fichier:
# lstm_model.h5

def build_model(train, n_input):
    # on fait appel à la fonction to_supervised pour augmanter la taille des données d'antrainement
    # et de restructurer train sous forme train_x, train_y, c'est pour ça la fonction est appellée to_supervised
    train_x, train_y = to_supervised(train, n_input)
    # la taille finale des données d'entrainement est (130 * 10)-10 = 1290, on divise 1290 sur batch_size
    # 1290/16 c'est le nombre de fois qu'on fait mise à jour des poids du RNN LSTM pour chaque epoch d'entrainement
    #verbose, epochs, batch_size = 0, 20, 16
    verbose, epochs, batch_size = 0, 24, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # définir le modèle
    model = Sequential()
    model.add(LSTM(180, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(180, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    opt = keras.optimizers.Adam(learning_rate=0.011)
    model.compile(loss='mse', optimizer=opt)
    #model.compile(loss='mse', optimizer='adam')
    # fit le modèle
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # sauvegarder le modèle dans le fichier 'lstm_model.h5'
    model.save('lstm_model.h5')
    return model

# cette fonction permet de faire la prédiction en utilisant le modèle déjà entrainé dans les paramètres
# history représent tous les vecteurs dans train, c'est une liste de vecteurs, pour faire la prédiction
# on prend juste les 10 dernières valeurs dans input_x: input_x = data[-n_input:, 0], ensuite, on utilise le 
# modèle entrainé pour faire la prédiction dans le vecteur yhat, qui a la form yhat.shape (10,1)

def forecast(model, history, n_input):
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	input_x = data[-n_input:, 0]
	input_x = input_x.reshape((1, len(input_x), 1))
	yhat = model.predict(input_x, verbose=0)
	yhat = yhat[0]
	return yhat

# la fonction evaluate_model permet d'evaluer le modèle LSTM, en utilisant test, en fait test est un vecteur de 
# 10 vecteurv, et chaque sous vecteur contient 10 valeurs, l'idée de cette fonction est d'utiliser les 10
# dernière valeurs de train c'est à dire le dernier vecteur dans train pour prédir les 10 premières valeurs de test 
# c'est à dire le premier vecteur dans test. Ensuite, elle va rajouter le vecteur prédit dans train 
# via l'instruction: history.append(test[i, :]) pour avoir un nouveau vecteur d'entrainement: history, ensuite elle
# utilise le dernier vecteur dans history, c'est à dire le vecteur prédit précedemment pour prédire le deuxième 
# vecteur dans test, jusqu'à arrivée au vecteur 10.

# les 10 vecteurs prédis sont sauvegardés dans la liste predictions, et test[i, :] représente les 10 vecteurs dans test
# qui sont considérés comme actuel, les 2 sont passés à la fonction: evaluate_forecasts pour calculer les RMSE
def evaluate_model(model, train, test, n_input):
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		yhat_sequence = forecast(model, history, n_input)
		predictions.append(yhat_sequence)
		history.append(test[i, :])
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# connecter au serveur influxdb
client = connect("10.111.219.45", 8086)

# récupération des données d'entrainement
data = get_training_data()

# répartir les données de dataset entre train(1300) et test(100), ensuite regrouper chaque 10 valeurs dans un vecteur 
# pour train et test: train(130, 10, 1) et test(10, 10, 1) 
#train, test = split_dataset(dataset.values)
train, test = split_dataset(data)
# evaluer le modèle et récupérer RMSE
n_input = timestep * 1
model = build_model(train, n_input)
score, scores = evaluate_model(model, train, test, n_input)
model.summary()
# afficher le RMSE
summarize_scores('lstm', score, scores)
# déssiner le RMSE en utilisant matplotlib
steps = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
#pyplot.plot(days, scores, marker='o', label='lstm')
#pyplot.show()
