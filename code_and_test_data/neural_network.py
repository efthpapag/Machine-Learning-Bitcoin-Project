import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from neural_network import *
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def neuralNetwork(data):
    arrD=[]
    arrP=[]
    for key in data.keys():
        #arrD.append([key])
        arrP.append(data[key])
    for x in range(1,1517):
        arrD.append([x])
    arrP_train, arrP_test = train_test_split(arrP, test_size=0.2, shuffle=False)
    trainBatches = []
    trainBatchesTarget = []
    i = 0
    while (i+50!= len(arrP_train)-1):
        l = []
        j = i
        while (j<i+50):
            l.append(arrP_train[j])
            j += 1
        trainBatches.append(l)
        trainBatchesTarget.append(arrP_train[i + 50])
        i += 1
    testBatches = []
    while (i+50!= len(arrP)-1):
        l = []
        j = i
        while (j<i+50):
            l.append(arrP[j])
            j += 1
        testBatches.append(l)
        i += 1
    neuralNetwork = MLPRegressor(hidden_layer_sizes=(50,12,))
    neuralNetwork.fit(trainBatches, trainBatchesTarget)
    predictions = neuralNetwork.predict(testBatches)
    rmse = mean_squared_error(arrP_test, predictions)
    r2 = r2_score(arrP_test, predictions)
    print(rmse)
    print(r2)
    arrD2=[]
    for i in range(0,len(predictions)):
        arrD2.append([i])
    #cross-validation
    mean_squared_errors = cross_val_score(neuralNetwork, trainBatches, trainBatchesTarget, cv=5, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(neuralNetwork, arrD, arrP, cv=5, scoring='r2')
    rmseCV = mean_squared_errors.mean()
    r2CV = r2_scores.mean()
    print(rmseCV)
    print(r2CV)

    plt.scatter(arrD2,arrP_test, s=10)
    plt.xlabel('dates')
    plt.ylabel('prices')
    plt.plot(arrD2, predictions, color='r')
    plt.show()