Prerequisites:

Microsoft Azure Subscription

Anaconda Jupyter Notebook

Python 3.6


-----------------------------------------------------

Imports:

import azureml.core
import pandas as pd
from azureml.core.workspace import Workspace
from azureml.train.automl.run import AutoMLRun
import time
import logging
from sklearn import datasets
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import random
import numpy as np

-------------------------------------------------------
Preprossing into Numpy arrays:

from sklearn.model_selection import train_test_split
# select all the Attributes which you include to train
X = data.drop(['FARE'],axis=1)
# select a target column
y=data['FARE']
# changing the data into Numpy arrays 
X=X.values
# changing the data into Numpy arrays
y=y.values
# Splitting the data set for further testing of algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

-------------------------------------------------------

For Further Prediction

result=fitted_model.predict(X_test)

-------------------------------------------------------

Plotting a graph:

import matplotlib.pyplot as plt
plt.plot(y_test)
plt.plot(result)
plt.rcParams['figure.figsize']=(20,10)
plt.show()
# Blue is actual
# yellow is predicted