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

## training the seasonal models

###### S U M M E R

dfSummer = pd.read_csv("allSummer_delhi.csv")

## defining variables for: SUMMER

XSummer = dfSummer[['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature']]
ySummer = dfSummer['O3P24']

print('Done defining summer variables')

## summer xgboost regressor
print('STARTING: SUMMER xgboost regressor')
t0 = time.time()

'''create regressor object'''
SummerxgbRegressor = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=300, verbosity = 0, random_state = 0, silent = True)

'''fitting regressor'''
SummerxgbRegressor.fit(XSummer, ySummer)

t1 = time.time()
xgbtotal = t1-t0

print("done training summer xgboost regressor | time taken: %f seconds" %xgbtotal)


###### M O N S O O N

dfMonsoon = pd.read_csv("allMonsoon_delhi.csv")


## defining variables for: MONSOON

XMonsoon = dfMonsoon[['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature']]
yMonsoon = dfMonsoon['O3P24']

print('Done defining monsoon variables')

## monsoon xgboost regressor
print('STARTING: MONSOON xgboost regressor')
t0 = time.time()

'''create regressor object'''
MonsoonxgbRegressor = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=300, verbosity = 0, random_state = 0, silent = True)

'''fitting regressor'''
MonsoonxgbRegressor.fit(XMonsoon, yMonsoon)

t1 = time.time()
xgbtotal = t1-t0

print("done training monsoon xgboost regressor | time taken: %f seconds" %xgbtotal)


###### F A L L

dfFall = pd.read_csv("allFall_delhi.csv")

## defining variables for: FALL

XFall = dfFall[['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature']]
yFall = dfFall['O3P24']

print('Done defining fall variables')

## fall xgboost regressor
print('STARTING: FALL xgboost regressor')
t0 = time.time()

'''create regressor object'''
FallxgbRegressor = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=300, verbosity = 0, random_state = 0, silent = True)

'''fitting regressor'''
FallxgbRegressor.fit(XFall, yFall)

t1 = time.time()
xgbtotal = t1-t0

print("done training fall xgboost regressor | time taken: %f seconds" %xgbtotal)


###### W I N T E R

dfWinter = pd.read_csv("allWinter_delhi.csv")


## defining variables for: WINTER

XWinter = dfWinter[['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature']]
yWinter = dfWinter['O3P24']

print('Done defining winter variables')

## winter xgboost regressor
print('STARTING: WINTER xgboost regressor')
t0 = time.time()

'''create regressor object'''
WinterxgbRegressor = xgb.XGBRegressor(learning_rate=0.1, max_depth=10, n_estimators=300, verbosity = 0, random_state = 0, silent = True)

'''fitting regressor'''
WinterxgbRegressor.fit(XWinter, yWinter)

t1 = time.time()
xgbtotal = t1-t0

print("done training winter xgboost regressor | time taken: %f seconds" %xgbtotal)

## defining the air quality index values

def AQIcalc(Ozone):
    if Ozone <= 104:
        return('Good')
    if 105 <= Ozone <=134:
        return('Moderate')
    if 135 <= Ozone <= 164:
        return('Unhealthy for Sensitive Populations')
    if 165 <= Ozone <= 204:
        return('Unhealthy')
    if 205 <= Ozone:
        return('Very Unhealthy')

## starting calculator

## enter date > choose season > 
print('>> Starting: 24-Hour Ozone Forecast Calculator <<')
month = int(input('What is the month of the current date? (Enter a number, 1-12) : '))
theList = ['Particulate Matter 10 (PM10)', 'Nitric Oxide (NO)', 'Nitrogen Dioxide (NO2)', 'Nitrogen Oxides (NOx)', 'Ammonia (NH3)', 'Carbon Monoxide (CO)', 'Sulfur Dioxide (SO2)', 'Current Ozone (O3)', 'Toluene', 'Xylene', 'Temperature']
predVars = []


if month == 3 or month == 4 or month == 5:
    print("That's in the Summer.")
    print('- - - - - - - - - - - - - - - - - - -')
    predVars = []
    
    for x in range(len(theList)):
        currVar = theList[x]
        currInput = (input('Enter %s : ' % currVar))
        predVars.append(float(currInput))
        
    df2Input = pd.DataFrame([predVars], columns =['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature'])
    yPred = SummerxgbRegressor.predict(df2Input)
    O3forecast = (round(yPred[0], 2))
    print('- - - - - - - - - - - - - - - - - - -')
    print('The ozone forecast for 24 hours is : ', O3forecast, ' µg/m³')
    AQI = AQIcalc(O3forecast)
    print('The Air Quality Index is : ', AQI)
        
    
elif month == 6 or month == 7 or month == 8 or month == 9:
    print("That's in Monsoon season.")
    predVars = []
    
    for x in range(len(theList)):
        currVar = theList[x]
        currInput = (input('Enter %s : ' % currVar))
        predVars.append(float(currInput))
        
    df2Input = pd.DataFrame([predVars], columns =['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature'])
    yPred = MonsoonxgbRegressor.predict(df2Input)
    O3forecast = yPred[0]
    print('The ozone forecast in 24 hours is : ', O3forecast)
    AQI = AQIcalc(O3forecast)
    print('The Air Quality Index is : ', AQI)

elif month == 10 or month == 11 or month == 12:
    print("That's in the Fall.")
    print('- - - - - - - - - - - - - - - - - - -')
    predVars = []
    
    for x in range(len(theList)):
        currVar = theList[x]
        currInput = (input('Enter %s : ' % currVar))
        predVars.append(float(currInput))
        
    df2Input = pd.DataFrame([predVars], columns =['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature'])
    yPred = FallxgbRegressor.predict(df2Input)
    O3forecast = (round(yPred[0], 2))
    print('- - - - - - - - - - - - - - - - - - -')
    print('The ozone forecast for 24 hours is : ', O3forecast, ' µg/m³')
    AQI = AQIcalc(O3forecast)
    print('The Air Quality Index is : ', AQI)
    
elif month == 1 or month == 2:
    print("That's in the Winter.")
    print('- - - - - - - - - - - - - - - - - - -')
    predVars = []
    
    for x in range(len(theList)):
        currVar = theList[x]
        currInput = (input('Enter %s : ' % currVar))
        predVars.append(float(currInput))
        
    df2Input = pd.DataFrame([predVars], columns =['PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Toluene', 'Xylene', 'Temperature'])
    yPred = WinterxgbRegressor.predict(df2Input)
    O3forecast = (round(yPred[0], 2))
    print('- - - - - - - - - - - - - - - - - - -')
    print('The ozone forecast for 24 hours is : ', O3forecast, ' µg/m³')
    AQI = AQIcalc(O3forecast)
    print('The Air Quality Index is : ', AQI)
    
else:
    print('Please try again with the month number, 1-12.')
