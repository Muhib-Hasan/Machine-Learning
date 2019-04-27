#-------------------------necessary library--------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score

FILEPATH = "dataset_v4.csv"

# Path of dataset files which are in csv
dataframe=pd.read_csv(FILEPATH)

#-----------------------retrieve data--------------------------------------------

features = ["Code","Year","Tempareture","Humidity","Rainfall","Wind Speed","Bright Sunshine","Cloud Coverage","Area"]
target = "Production"

x = dataframe[features]
y = dataframe[target]
print(x.shape, y.shape)

#--------------------Train-test data---------------------------------

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.30, random_state=0)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#--------------------SVR Model--------------------------------

regressor = SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
  gamma='auto_deprecated', kernel='linear', max_iter=-1, shrinking=True,
  tol=0.001, verbose=False)
regressor.fit(x_train, y_train)

#-------------------Model Evaluation---------------------------------

pred = regressor.predict(x_test)
print(regressor.score(x_test, y_test))
print(r2_score(y_test,pred))

#----------------------------------------------------