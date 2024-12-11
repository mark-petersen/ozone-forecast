## WINTER seasonal model code
## importing libraries

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import time
from sklearn import preprocessing
import seaborn as sns


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.metrics
import math
from numpy import mean
from numpy import std

from sklearn.metrics import mean_squared_error, r2_score

df2 = pd.read_csv('allWinter_delhi.csv')
df2.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df2.drop(["a"], axis=1, inplace=True)
df2.drop(["index"], axis=1, inplace=True)

print('Done importing columns')
print(df2.columns, len(df2))

# extra nan check:

print(df2.isna().sum().sum())
df1 = df2[df2.isna().any(axis=1)]
print(df1)

print(np.mean(df2['PM10']))

## defining variables

#X = df2[['PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3', 'Toluene', 'Temp']] # , 'Humid', Xylene
X = df2[['PM10', 'NO', 'NO2', 'NH3','CO', 'SO2', 'O3', 'Toluene','Xylene', 'Temperature']]

y = df2['O3P24']

from sklearn.model_selection import train_test_split
X_train, X_test1, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.1)

print("Size of training dataset: {} rows".format(X_train.shape[0]))
print("Size of testing dataset: {} rows".format(X_test1.shape[0]))

print('Done defining variables')

## scaling variables

'''importing libraries'''
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

scaler = preprocessing.RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled1 = scaler.fit_transform(X_test1)

print('Done scaling :)')

## linear regression
t0 = time.time()
print("STARTING: Linear Regression")

'''importing library'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

'''create linear regressor object'''
lnRegressor = LinearRegression()

'''fitting regressor'''
lnRegressor.fit(X_train, y_train)

# cv stuff

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

cvln_r2scores = cross_val_score(lnRegressor, X_train, y_train, cv = 10, scoring = 'r2')
print(cvln_r2scores)
cvln_rmsescores = cross_val_score(lnRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvln_rmsescores)
cvln_maescores = cross_val_score(lnRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvln_maescores)

# calculating r^2 adj score

cvln_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvln_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvln_adj.append(adj_r2)    

cvln_r2score = round((np.mean(cvln_r2scores)), 4)
cvln_adjscore = round((np.mean(cvln_adj)), 4)
cvln_rmsescore = int(round(abs(np.mean(cvln_rmsescores))))
cvln_maescore = int(round(abs(np.mean(cvln_maescores))))


print(cvln_r2score, cvln_adjscore, cvln_rmsescore, cvln_maescore)

t1 = time.time()
lntotal = t1-t0

print("done training linear regressor | time taken: %f seconds" %lntotal)
yPredln = lnRegressor.predict(X_test1)
print('The R^2 value for linear Regressor is :', round((r2_score(y_test, yPredln)), 4))

## kneighbors regression
print("STARTING: KN")

t0 = time.time()

'''importing library'''
from sklearn.neighbors import KNeighborsRegressor

'''create regressor object'''
knRegressor = KNeighborsRegressor(n_neighbors = 4, metric = 'minkowski', p = 1)

'''fitting regressor'''
knRegressor.fit(X_train_scaled, y_train)

# cv stuff

cvkn_r2scores = cross_val_score(knRegressor, X_train, y_train, cv = 10, scoring = 'r2')
print(cvkn_r2scores)
cvkn_rmsescores = cross_val_score(knRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvkn_rmsescores)
cvkn_maescores = cross_val_score(knRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvkn_maescores)

# calculating r^2 adj score

cvkn_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvkn_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvkn_adj.append(adj_r2)    

cvkn_r2score = round((np.mean(cvkn_r2scores)), 4)
cvkn_adjscore = round((np.mean(cvkn_adj)), 4)
cvkn_rmsescore = int(round(abs(np.mean(cvkn_rmsescores))))
cvkn_maescore = int(round(abs(np.mean(cvkn_maescores))))

print(cvkn_r2score, cvkn_adjscore, cvkn_rmsescore, cvkn_maescore)


t1 = time.time()
kntotal = t1-t0

print("done training knn regressor | time taken: %f seconds" %kntotal)

yPredkn = knRegressor.predict(X_test_scaled1)
print('The R^2 value for KNN Regressor is :', round((r2_score(y_test, yPredkn)), 4))


## svr regressor
print("STARTING: SVR")

t0 = time.time()

'''importing library'''
from sklearn.svm import SVR

'''create regressor object'''
svrRegressor = SVR(C = 10, gamma = 0.1, kernel= 'rbf')

'''fitting regressor'''
svrRegressor.fit(X_train_scaled, y_train)

# cv stuff

cvsvr_r2scores = cross_val_score(svrRegressor, X_train_scaled, y_train, cv = 10, scoring = 'r2')
print(cvsvr_r2scores)
cvsvr_rmsescores = cross_val_score(svrRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvsvr_rmsescores)
cvsvr_maescores = cross_val_score(svrRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvsvr_maescores)


# calculating r^2 adj score

cvsvr_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvsvr_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvsvr_adj.append(adj_r2)

cvsvr_r2score = round((np.mean(cvsvr_r2scores)), 4)
cvsvr_adjscore = round((np.mean(cvsvr_adj)), 4)
cvsvr_rmsescore = int(round(abs(np.mean(cvsvr_rmsescores))))
cvsvr_maescore = int(round(abs(np.mean(cvsvr_maescores))))

print(cvsvr_r2score, cvsvr_adjscore, cvsvr_rmsescore, cvsvr_maescore)

t1 = time.time()
svrtotal = t1-t0

print("done training svr regressor | time taken: %f seconds" %svrtotal)

yPredsvr = svrRegressor.predict(X_test_scaled1)
print('The R^2 value for SVR Regressor is :', round((r2_score(y_test, yPredsvr)), 4))

## defining variables WITH MULTICOLLINEARITY: random forest, decision tree, maybe adaboost, and xgboost

X = df2[['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature']]
#X = df2[['PM10', 'NO', 'NO2', 'NH3','CO', 'SO2', 'O3', 'Toluene','Xylene', 'Temperature']]

y = df2['O3P24']

X_train, X_test2, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.1)
2
print("Size of training dataset: {} rows".format(X_train.shape[0]))
print("Size of testing dataset: {} rows".format(X_test2.shape[0]))

print('Done defining NEW variables')

## random forest regressor
print("STARTING: random forest")
t0 = time.time()

'''importing library'''
from sklearn.ensemble import RandomForestRegressor

'''create regressor object'''
rfRegressor = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=250) 

'''fitting regressor'''
rfRegressor.fit(X_train, y_train)

# cv stuff

cvrf_r2scores = cross_val_score(rfRegressor, X_train, y_train, cv = 10, scoring = 'r2') # x_train was scaled for some reason, check if it changes things?
print(cvrf_r2scores)
cvrf_rmsescores = cross_val_score(rfRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvrf_rmsescores)
cvrf_maescores = cross_val_score(rfRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvrf_maescores)


# calculating r^2 adj score

cvrf_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvrf_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvrf_adj.append(adj_r2)

cvrf_r2score = round((np.mean(cvrf_r2scores)), 4)
cvrf_adjscore = round((np.mean(cvrf_adj)), 4)
cvrf_rmsescore = int(round(abs(np.mean(cvrf_rmsescores))))
cvrf_maescore = int(round(abs(np.mean(cvrf_maescores))))

print(cvrf_r2score, cvrf_adjscore, cvrf_rmsescore, cvrf_maescore)

print(cvrf_r2score, cvrf_adjscore)

t1 = time.time()
rftotal = t1-t0

print("done training random forest regressor | time taken: %f seconds" %rftotal)
yPredrf = rfRegressor.predict(X_test2)
print('The R^2 value for Random Forest Regressor is :', round((r2_score(y_test, yPredrf)), 4))

## decision tree regressor
print('STARTING: dt')
t0 = time.time()

'''importing library'''
from sklearn.tree import DecisionTreeRegressor

'''create regressor object'''
dtRegressor = DecisionTreeRegressor(random_state=0, max_depth = 6)

'''fitting regressor'''
dtRegressor.fit(X_train,y_train)

# cv stuff

cvdt_r2scores = cross_val_score(dtRegressor, X_train, y_train, cv = 10, scoring = 'r2')
print(cvdt_r2scores)
cvdt_rmsescores = cross_val_score(dtRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvdt_rmsescores)
cvdt_maescores = cross_val_score(dtRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvdt_maescores)


# calculating r^2 adj score

cvdt_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvdt_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvdt_adj.append(adj_r2)

cvdt_r2score = round((np.mean(cvdt_r2scores)), 4)
cvdt_adjscore = round((np.mean(cvdt_adj)), 4)
cvdt_rmsescore = int(round(abs(np.mean(cvdt_rmsescores))))
cvdt_maescore = int(round(abs(np.mean(cvdt_maescores))))

print(cvdt_r2score, cvdt_adjscore, cvdt_rmsescore, cvdt_maescore)

t1 = time.time()
dttotal = t1-t0

print("done training decision tree regressor | time taken: %f seconds" %dttotal)

yPreddt = dtRegressor.predict(X_test2)
print('The R^2 value for Decision Tree Regressor is :', round((r2_score(y_test, yPreddt)), 4))

## adaboost regressor
print('STARTING: adaboost regressor')
t0 = time.time()

'''importing library'''
from sklearn.ensemble import AdaBoostRegressor

'''create regressor object'''
adaRegressor = AdaBoostRegressor(random_state=0, learning_rate = 0.1, n_estimators=100)

'''fitting regressor'''
adaRegressor.fit(X_train, y_train)

# cv stuff

cvada_r2scores = cross_val_score(adaRegressor, X_train, y_train, cv = 10, scoring = 'r2')
print(cvada_r2scores)
cvada_rmsescores = cross_val_score(adaRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvada_rmsescores)
cvada_maescores = cross_val_score(adaRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvada_maescores)

# calculating r^2 adj score

cvada_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvada_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvada_adj.append(adj_r2)

cvada_r2score = round((np.mean(cvada_r2scores)), 4)
cvada_adjscore = round((np.mean(cvada_adj)), 4)
cvada_rmsescore = int(round(abs(np.mean(cvada_rmsescores))))
cvada_maescore = int(round(abs(np.mean(cvada_maescores))))

print(cvada_r2score, cvada_adjscore, cvada_rmsescore, cvada_maescore)

t1 = time.time()
adatotal = t1-t0

print("done training Adaboost regressor | time taken: %f seconds" %adatotal)

yPredada = adaRegressor.predict(X_test2)
print('The R^2 value for Adaboost Regressor is :', round((r2_score(y_test, yPredada)), 4))

## xgboost regressor
print('STARTING: xgboost regressor')
t0 = time.time()

'''importing library'''
import xgboost as xgb

'''create regressor object'''
xgbRegressor = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=300, verbosity = 0, random_state = 0, silent = True)

'''fitting regressor'''
xgbRegressor.fit(X_train, y_train)

# cv stuff

cvxgb_r2scores = cross_val_score(xgbRegressor, X_train, y_train, cv = 10, scoring = 'r2')
print(cvxgb_r2scores)
cvxgb_rmsescores = cross_val_score(xgbRegressor, X_train, y_train, cv = 10, scoring = 'neg_root_mean_squared_error')
print(cvxgb_rmsescores)
cvxgb_maescores = cross_val_score(xgbRegressor, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
print(cvxgb_maescores)


# calculating r^2 adj score

cvxgb_adj = []

n = len(X_train)
k = len(X_train.columns)
for r in cvxgb_r2scores:
    adj_r2 = 1-(((1-r)*(n-1))/(n-k-1))
    cvxgb_adj.append(adj_r2)

cvxgb_r2score = round((np.mean(cvxgb_r2scores)), 4)
cvxgb_adjscore = round((np.mean(cvxgb_adj)), 4)
cvxgb_rmsescore = int(round(abs(np.mean(cvxgb_rmsescores))))
cvxgb_maescore = int(round(abs(np.mean(cvxgb_maescores))))

print(cvxgb_r2score, cvxgb_adjscore, cvxgb_rmsescore, cvxgb_maescore)

t1 = time.time()
xgbtotal = t1-t0

print("done training xgboost regressor | time taken: %f seconds" %xgbtotal)

yPredxgb = xgbRegressor.predict(X_test2)
print('The R^2 value for xgboost Regressor is :', round((r2_score(y_test, yPredxgb)), 4))

##### printing cross-validations scores
# printing cross-validaion r^2 scores

print('The R^2 value for Linear Regression is         :', cvln_r2score)
print('The R^2 value for KNN Regressor is             :', cvkn_r2score)
print('The R^2 value for SVM Regressor is             :', cvsvr_r2score)
print('The R^2 value for Random Forests Regressor is  :', cvrf_r2score)
print('The R^2 value for Decision Tree Regressor is   :', cvdt_r2score)
print('The R^2 value for AdaBoost Regressor is        :', cvada_r2score)
print('The R^2 value for XGBoost Regressor is         :', cvxgb_r2score)

# printing cross-validation adjusted r^2 scores

print('The Adj. R^2 value for Linear Regression is        :', cvln_adjscore)
print('The Adj. R^2 value for KNN Regressor is            :', cvkn_adjscore)
print('The Adj. R^2 value for SVM Regressor is            :', cvsvr_adjscore)
print('The Adj. R^2 value for Random Forests Regressor is :', cvrf_adjscore)
print('The Adj. R^2 value for Decision Tree Regressor is  :', cvdt_adjscore)
print('The Adj. R^2 value for AdaBoost Regressor is       :', cvada_adjscore)
print('The Adj. R^2 value for XGBoost Regressor is        :', cvxgb_adjscore)

# printing cross-validation rmse scores

print('The RSME value for Linear Regression is         :', cvln_rmsescore)
print('The RSME value for KNN Regressor is             :', cvkn_rmsescore)
print('The RSME value for SVM Regressor is             :', cvsvr_rmsescore)
print('The RSME value for Random Forests Regressor is  :', cvrf_rmsescore)
print('The RSME value for Decision Tree Regressor is   :', cvdt_rmsescore)
print('The RSME value for AdaBoost Regressor is        :', cvada_rmsescore)
print('The RSME value for XGBoost Regressor is         :', cvxgb_rmsescore)

# printing cross-validation mae scores

print('The MAE value for Linear Regression is        :', cvln_maescore)
print('The MAE value for KNN Regressor is            :', cvkn_maescore)
print('The MAE value for SVM Regressor is            :', cvsvr_maescore)
print('The MAE value for Random Forests Regressor is :', cvrf_maescore)
print('The MAE value for Decision Tree Regressor is  :', cvdt_maescore)
print('The MAE value for AdaBoost Regressor is       :', cvada_maescore)
print('The MAE value for XGBoost Regressor is        :', cvxgb_maescore)

# printing time taken
print('The time taken for Linear Regression is         :', lntotal)
print('The time taken for KNN Regression is            :', kntotal)
print('The time taken for SVM Regression is            :', svrtotal)
print('The time taken for Random Forests Regression is :', rftotal)
print('The time taken for Decision Tree Regression is  :', dttotal)
print('The time taken for AdaBoost Regression is       :', adatotal)
print('The time taken for XGBoost Regression is        :', xgbtotal)

### TESTING

yPredln = lnRegressor.predict(X_test1)
yPredkn = knRegressor.predict(X_test_scaled1)
yPredsvr = svrRegressor.predict(X_test_scaled1)
yPredrf = rfRegressor.predict(X_test2)
yPreddt = dtRegressor.predict(X_test2)
yPredada = adaRegressor.predict(X_test2)
yPredxgb = xgbRegressor.predict(X_test2)

predR2ln = str((round((math.sqrt(r2_score(y_test,yPredln))), 3)))
predR2kn = str((round((math.sqrt(r2_score(y_test,yPredkn))), 3)))
predR2svr = str((round((math.sqrt(r2_score(y_test,yPredsvr))), 3)))
predR2rf = str((round((math.sqrt(r2_score(y_test,yPredrf))), 3)))
predR2dt = str((round((math.sqrt(r2_score(y_test,yPreddt))), 3)))
predR2ada = str((round((math.sqrt(r2_score(y_test,yPredada))), 3)))
predR2xgb = str((round((math.sqrt(r2_score(y_test,yPredxgb))), 3)))


print(predR2kn)
print(predR2rf)
print(predR2dt)
print(predR2xgb)

# ln reg

print('The R^2 value for Linear Regression is   :', round((r2_score(y_test,yPredln)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPredln, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])

plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter Linear Regression - Correlation: %s" %predR2ln, size = 20)
plt.savefig('figs/Winter Linear Regression.png')
plt.show()


# kn reg
print('The R^2 value for KNN Regression is   :', round((r2_score(y_test,yPredkn)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPredkn, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])

plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter KNN Regression - Correlation: %s" %predR2kn, size = 20)
plt.savefig('figs/Winter KNN Regression.png')
plt.show()


# svr reg
print('The R^2 value for SVM Regression is   :', round((r2_score(y_test,yPredsvr)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPredsvr, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])


plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter SVM Regression - Correlation: %s" %predR2svr, size = 20)
plt.savefig('figs/Winter SVM Regression.png')
plt.show()


# rf reg
print('The R^2 value for Random Forests Regression is   :', round((r2_score(y_test,yPredrf)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPredrf, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])

plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter Random Forests Regression - Correlation: %s" %predR2rf, size = 20)
plt.savefig('figs/Winter Random Forests Regression.png')
plt.show()


# dt reg
print('The R^2 value for Decision Tree Regression is   :', round((r2_score(y_test,yPreddt)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPreddt, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])

plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter Decision Tree Regression - Correlation: %s" %predR2dt, size = 20)
plt.savefig('figs/Winter Decision Tree Regression.png')
plt.show()


# ada reg
print('The R^2 value for AdaBoost Regression is   :', round((r2_score(y_test,yPredada)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPredada, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])

plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter AdaBoost Regression - Correlation: %s" %predR2ada, size = 20)
plt.savefig('figs/Winter AdaBoost Regression.png')
plt.show()


# xgb reg
print('The R^2 value for XGBoost Regression is   :', round((r2_score(y_test,yPredxgb)), 3))
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, yPredxgb, s= 4)
x = np.linspace(0,200,200)
y = 1*x+0
plt.plot(x, y, '-r', label='y=2x+1')
plt.xlim([0, 200])
plt.ylim([0, 200])

plt.xlabel('Actual O3 Concentration (µg/m³)')
plt.ylabel('Predicted O3 Concentration (µg/m³)')
plt.grid(b= True)
plt.title("Winter XGBoost Regression - Correlation: %s" %predR2xgb, size = 20)
plt.savefig('figs/Winter XGBoost Regression.png')
plt.show()


from IPython.display import display, HTML

#create a function that calculates the percentage between two values.

def percentDiff(a, f): # a is actual, f is forecast #instead do prediction - observed (makes more sense )
    try:
        return((abs((a)-f)/a)*100)
    
    except ZeroDivisionError:
        return float(0)

def pre_post_difference(a, b):
    return((a - b))
    
df3 = pd.Series.to_frame(y_test)
#df3 = df3.reset_index(drop=True)
df3.rename(columns={df3.columns[0]: "Actual O3" }, inplace = True)
df3['Predicted O3'] = yPredkn
df3['Amount Different'] = pre_post_difference(df3['Actual O3'], df3['Predicted O3'])
df3['Percent Error'] = round(percentDiff(df3['Actual O3'], df3['Predicted O3']), 2)

df3.hist(column='Winter Percent Different', grid=True, bins=50, range=[-100, 100])#xlabelsize=None, ylabelsize=None, figsize=None, 

## plot of the amount different
fig = plt.figure(figsize=(30, 16))
plo = df3.hist(column='Amount Different',grid=True, bins=51, figsize=(8, 5), range=[6.5, 12.5])
plo[0][0].set_ylabel('Frequency', size = 10)
plo[0][0].set_xlabel('Amount Different from Acutal (µ O₃/m³)', size = 10)

'''
calculating AQI

'''
'''
0 : GOOD                : 0-105
1 : MODERATE            : 105-135
2 : UNHEALTHY FOR SENS. : 135-165
3 : UNHEALTHY           : 165-205
4 : VERY UNHEALTHY      : 205-380

df['employrate'] = np.where(
   (df['employrate'] <=55) & (df['employrate'] > 50) , 11, df['employrate']
   )

'''

df3['Actual AQI'] = df3['Actual O3']
#df['B'] = np.where(df['B'].between(8,11), 0, df['B'])

df3['Actual AQI'] = (np.where(df3['Actual AQI'].between(0,105), 0, df3['Actual AQI']))
df3['Actual AQI'] = (np.where(df3['Actual AQI'].between(105,135), 1, df3['Actual AQI']))
df3['Actual AQI'] = (np.where(df3['Actual AQI'].between(135,165), 2, df3['Actual AQI']))
df3['Actual AQI'] = (np.where(df3['Actual AQI'].between(165,205), 3, df3['Actual AQI']))
df3['Actual AQI'] = (np.where(df3['Actual AQI'].between(205,350), 4, df3['Actual AQI']))
df3['Actual AQI'] = (np.where(df3['Actual AQI'].between(350,500), 5, df3['Actual AQI']))


df3['Predicted AQI'] = df3['Predicted O3']

df3['Predicted AQI'] = (np.where(df3['Predicted AQI'].between(0,105), 0, df3['Predicted AQI']))
df3['Predicted AQI'] = (np.where(df3['Predicted AQI'].between(105,135), 1, df3['Predicted AQI']))
df3['Predicted AQI'] = (np.where(df3['Predicted AQI'].between(135,165), 2, df3['Predicted AQI']))
df3['Predicted AQI'] = (np.where(df3['Predicted AQI'].between(165,205), 3, df3['Predicted AQI']))
df3['Predicted AQI'] = (np.where(df3['Predicted AQI'].between(205,350), 4, df3['Predicted AQI']))
df3['Predicted AQI'] = (np.where(df3['Predicted AQI'].between(350,500), 5, df3['Predicted AQI']))


df3['Amount Diff AQI'] = pre_post_difference(df3['Actual AQI'], df3['Predicted AQI'])
df3['Percent Error AQI'] = round(percentDiff(df3['Actual AQI'], df3['Predicted AQI']), 5)

display(df3)

print(df3['Actual AQI'].value_counts(sort=True))
print(df3['Predicted AQI'].value_counts(sort=True))
print(df3['Amount Diff AQI'].value_counts(sort=True))
print(len(df3))

