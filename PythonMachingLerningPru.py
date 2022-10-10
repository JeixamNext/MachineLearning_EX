"""
Created: 20-04-2022

@Author: Jaime Rodriguez Collado
"""

from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
#----------------------------------------------------------------------------------------------------------------------------------------


def decisioncsv():
    data_animals=pd.read_csv("market-prices-animal-products.csv")

    print (data_animals.head())

    x=data_animals[["Period","MP Market Price"]]
    y=data_animals["Product desc"]
    #realiza un test de entrenamiento para los datos validos 
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2,test_size=0.25)
    #el argritmo se basa en los datos de entrenamiento
    algoritmo= tree.DecisionTreeClassifier().fit(x_train,y_train)
    # tets es la regla y train es la predicion basada en los datos de entrnamiento
    true=y_test
    predicted=algoritmo.predict(x_test)
    predictions=pd.DataFrame({"true":true, "predicted":predicted})
    print(predictions)

def predicion(args,cabecera,valorA,valorB):
    data_args=pd.read_excel(args)

    print (data_args.head(10))

    x=data_args[[valorA,valorB]]
    y=data_args[cabecera]
    #realiza un test de entrenamiento para los datos validos 
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2,test_size=0.33)
    #el argritmo se basa en los datos de entrenamiento
    algoritmo= tree.DecisionTreeClassifier().fit(x_train,y_train)
    # tets es la regla y train es la predicion basada en los datos de entrnamiento
    true=y_test
    predicted=algoritmo.predict(x_test)
    predictions=pd.DataFrame({"true":true, "predicted":predicted})
    print(predictions)

def pru(args):
    data_args=pd.read_csv(args)
    #remplaza los datos en blanco
    data_args.replace(np.nan,"0")
    #trasforma los datos string en numerico y crea unas columnas
    encoder=LabelEncoder()
    data_args["plataforma"]=encoder.fit_transform(data_args.Platform.values)
    data_args["publicacion"]=encoder.fit_transform(data_args.Publisher.values)
    data_args["genero"]=encoder.fit_transform(data_args.Genre.values)
    #datos de alimentacion de la red neuronal
    x=data_args[["plataforma","Global_Sales","publicacion","genero"]]
    #salida de la red neuronal
    y=data_args["Genre"]
    #estandarizacion de un conjunto de datos mediante la desviacion estandar eliminando la media
    #Estimador comun de proceso de aprendizaje automatico
    #Sistema de entrenamiento y test
    X_train, X_test, y_train, y_test=train_test_split(x, y,random_state=2,test_size=0.33,train_size=0.33)
    #preprocesamiento, estandarizacion de un conjunto de datos con la desviacion estandar
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #clasificador de red neuronal, define la red neuronal
    # hidden_layer_sizes:numero de neuronas en la capa oculta
    #adam es la funcion de resolucion de la red (hay varias segun necesidades)
    #es importante ajustar la red para obtener el mejor resultado posible en cada caso 
    RedNeural=MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001,
                         solver='adam', random_state=21,tol=0.000000001)
    #RedNeural = MLPClassifier(hidden_layer_sizes=(6,6,6,6),solver='lbfgs',max_iter=6000)
    
    #hacer prediciones y entrenar la red
    RedNeural.fit(X_train,y_train)
    # predicion mediante matriz de confusion (confusion_matrix)
    prediction=RedNeural.predict(X_test)
    #Ver los resultados
    print(classification_report(y_test,prediction))
    
print("que desea hacer?")
res=input()
if res=="predicioncsv":
    decisioncsv()
elif res=="prediccion":
    predicion("frutas.xlsx","nombre","size","precio")
elif res=="red neural":
    pru("vgsales2.csv")