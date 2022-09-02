import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns,set()

def logisticRegression(data):
    arrD=[]
    arrP=[]
    avg20=[]
    for key in data.keys():
        #arrD.append([key])
        arrP.append([data[key]])
    for x in range(1,1517):
        arrD.append([x])
    for i in range(0,1516):
        if (i < 20):
            avg20.append([0])
        else:
            L = []
            for j in range(i-20, i):
                L.append(arrP[j][0])
            avgL = 0
            for k in L:
                avgL += k
            avgL = avgL / 20
            if (arrP[i][0] >= avgL):
                avg20.append([1])
            else:
                avg20.append([0])
    lr_model = LogisticRegression()
    lr_model.fit(arrD, avg20)
    y_pred = lr_model.predict(arrD)
    print(y_pred)
    #Actual value and the predicted value
    #rmse = mean_squared_error(avg20, y_pred)
    #r2 = r2_score(avg20, y_pred)
    #from sklearn import metrics
    #cross-validation
    #ac = cross_val_score(lr_model, arrD, avg20, cv=5, scoring='acuracy')
    #print(ac)
    from sklearn.metrics import classification_report, confusion_matrix
    matrix = confusion_matrix(avg20, y_pred)
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(classification_report(avg20, y_pred))