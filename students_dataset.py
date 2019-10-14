import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
dataset= pd.read_csv("student-mat.csv")
X= dataset.iloc[:,:-3].values
y= dataset.iloc[:,32].values
dataframe= pd.DataFrame(X)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()  
#To Encode the data with two variables
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
X[:, 8] = labelencoder.fit_transform(X[:, 8])
X[:, 9] = labelencoder.fit_transform(X[:, 9])
X[:, 10] = labelencoder.fit_transform(X[:,10])
X[:, 11] = labelencoder.fit_transform(X[:,11 ])
X[:, 12] = labelencoder.fit_transform(X[:, 12])
X[:, 13] = labelencoder.fit_transform(X[:, 13])
X[:, 14] = labelencoder.fit_transform(X[:, 14])
X[:, 15] = labelencoder.fit_transform(X[:, 15])
X[:, 16] = labelencoder.fit_transform(X[:, 16])
#to Encode the data with more than three Variables
onehotencoder = OneHotEncoder(categorical_features = [5]) 
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [7])
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [8])
X = onehotencoder.fit_transform(X).toarray()

# to avoid the Dummy Variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 1)#using the Modeled dataset for Training and Testing

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test results
y_pred = regressor.predict(X_test)

import statsmodels.regression.linear_model  as sm #Automate the Backward Elimination process
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]]
X_Modeled = backwardElimination(X_opt, SL)

