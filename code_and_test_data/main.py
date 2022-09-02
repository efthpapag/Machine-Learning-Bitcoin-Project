import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import *
from neural_network import *
from logistic_regression import *


def dataFormatting():
    dataset = pd.read_csv("BTC-USD.csv")

    dataset.drop('Open', axis = 1, inplace = True)
    dataset.drop('High', axis = 1, inplace = True)
    dataset.drop('Low',  axis = 1, inplace = True)
    dataset.drop('Adj Close',  axis = 1, inplace = True)
    dataset.drop('Volume',  axis = 1, inplace = True)
    data = dataset.set_index('Date').to_dict()
    data = data["Close"]
    #print(data)
    data = normalization(data)
    #visualization(data)
    return data

def normalization(data):
    all_values = data.values()
    max_value = max(all_values)
    min_value = min(all_values)
    for key in data:
        data[key] = (data[key] - min_value) / (max_value - min_value)
    return data

def visualization(data):
    plt.scatter(data.keys(),data.values(),s=10)
    plt.xlabel('dates')
    plt.ylabel('prices')
    plt.show()




def main():
    data = dataFormatting()
    choice = input(" 1. linear regression \n 2. logistic regression \n 3. neural network \n")
    if choice == "1":
        linearRegression(data)
    elif choice == "2":
        logisticRegression(data)
    elif choice == "3":
        neuralNetwork(data)
    else :
        print("Error")
main()