import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def linearRegression(data):
    arrD=[]
    arrP=[]
    for key in data.keys():
        #arrD.append([key])
        arrP.append([data[key]])
    for x in range(1,1517):
        arrD.append([x])
    regression_model = LinearRegression()
    regression_model.fit(arrD,arrP)
    y_predicted = regression_model.predict(arrD)
    rmse = mean_squared_error(arrP, y_predicted)
    r2 = r2_score(arrP, y_predicted)
    print('Slope:' ,regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    plt.scatter(arrD,arrP, s=10)
    plt.xlabel('dates')
    plt.ylabel('prices')
    # predicted values
    plt.plot(arrD, y_predicted, color='r')
    plt.show()
    #cross-validation
    mean_squared_errors = cross_val_score(regression_model, arrD, arrP, cv=5, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(regression_model, arrD, arrP, cv=5, scoring='r2')
    predictions = regression_model.predict(arrD)
    rmseCV = mean_squared_errors.mean()
    r2CV = r2_scores.mean()
    print('Root mean squared error in cross validation: ', rmseCV)
    print('R2 score in cross validation: ', r2CV)
    plt.scatter(arrD,arrP, s=10)
    plt.xlabel('dates')
    plt.ylabel('prices')
    plt.plot(arrD, predictions, color='r')
    plt.show()