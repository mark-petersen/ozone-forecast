# import libraries 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import time
from sklearn import preprocessing

from scipy.stats import loguniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import mean_squared_error, r2_score
import time
print('done importing libraries')

# load dataset

df2 = pd.read_csv("fixed_delhi_pollutant1.csv")
df2.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df2.drop(["a"], axis=1, inplace=True)
df = pd.read_csv("fixed_api_delhi_2015.csv")

Ozone = df2['O3']


OzoneP24 = Ozone.shift(24)
OzoneP24 = OzoneP24.replace(np.nan, 54.94)
df2['O3P24'] = OzoneP24

# importing weather ones

df2['Temp'] = df['Temperature']
df2['Humid'] = df['Relative Humidity']
df2['Cloud'] = df['Cloud Cover']
df2['Press'] = df['Sea Level Pressure']

# print done
print('done importing columns')
print(df2.columns, len(df2))

# split into input and output elements

X = df2[['PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temp', 'Humid']]

y = df2['O3P24']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.1)

print("Size of train dataset: {} rows".format(X_train.shape[0]))
print("Size of test dataset: {} rows".format(X_test.shape[0]))

# RUN TUNING AND TRAINING ON X_train, y_train <<<< IMPORTANT

# scaling
from sklearn.preprocessing import RobustScaler

scaler = preprocessing.RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print('done scaling')

### random forest

t0 = time.time()

# define model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space

tuned_parameters = [{'max_depth': [5,10, 15, 20, 50, 70], 'n_estimators': [10, 25, 50, 100,150, 200, 250]}]

# define search
search = GridSearchCV(model, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=2, cv=cv)

# execute search
result = search.fit(X_train, y_train)

t1 = time.time()
total = t1-t0

# summarize result
print('best score: %s' % result.best_score_)
print('best hyperparameters: %s' % result.best_params_)
print('time taken: %f seconds' %total)

### decision tree

t0 = time.time()

# define model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space

space = [{'max_depth': [1,2,3,4,5,10, 15, 20, 25, 50, 100,200]}]

# define search
search = GridSearchCV(model, space, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X_train, y_train)

t1 = time.time()
total = t1-t0

# summarize result
print('best score: %s' % result.best_score_)
print('best hyperparameters: %s' % result.best_params_)
print('time taken: %f seconds' %total)

### adaboost

t0 = time.time()

# define model
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space

tuned_parameters = [{'learning_rate': [0.1,1,2,3,4,5], 'n_estimators': [100,200,300, 400, 500]}]

# define search
search = GridSearchCV(model, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X_train, y_train)

t1 = time.time()
total = t1-t0

# summarize result
print('best score: %s' % result.best_score_)
print('best hyperparameters: %s' % result.best_params_)
print('time taken: %f seconds' %total)


### knn

t0 = time.time()

# define model
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space

tuned_parameters = [{'n_neighbors': [1,2,3,4,5,10,15,20], 'p': [1,2]}]

# define search
search = GridSearchCV(model, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X_train_scaled, y_train)

t1 = time.time()
total = t1-t0

# summarize result
print('best score: %s' % result.best_score_)
print('best hyperparameters: %s' % result.best_params_)
print('time taken: %f seconds' %total)

### svr 

t0 = time.time()

# define model
from sklearn.svm import SVR

model = SVR()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space

tuned_parameters = [{'kernel': ['linear', 'rbf', 'poly'], 'C':[1, 2, 3, 5, 6, 7, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}]

# define search
search = GridSearchCV(model, tuned_parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X_train_scaled, y_train)

t1 = time.time()
total = t1-t0

# summarize result
print('best score: %s' % result.best_score_)
print('best hyperparameters: %s' % result.best_params_)
print('time taken: %f seconds' %total)


### xgboost

t0 = time.time()

# define model
import xgboost as xgb

model = xgb.XGBRegressor()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space

tuned_parameters = [{'max_depth': [5,10, 15, 20, 25, 30],'learning_rate':[0.001, 0.01, 0.1, 0.5], 'n_estimators': [100,150,200, 250, 300]}]

# define search
search = RandomizedSearchCV(model, tuned_parameters, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)

# execute search
result = search.fit(X_train, y_train)

t1 = time.time()
total = t1-t0

# summarize result
print('best score: %s' % result.best_score_)
print('best hyperparameters: %s' % result.best_params_)
print('time taken: %f seconds' %total)

## NOTE : parts of this code were borrowed or modified from Jason Brownlee's article on hypertuning at https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
