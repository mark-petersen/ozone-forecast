import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("fixed_api_delhi_2015.csv")
df2 = pd.read_csv("fixed_delhi_pollutant1.csv")
df2.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df2.drop(["a"], axis=1, inplace=True)

df2 = pd.read_csv("fixed_delhi_pollutant1.csv")
df2.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df2.drop(["a"], axis=1, inplace=True)

# df weather columns

Date = df['Date']
Time = df['Time']
Temp = df['Temperature']
Humidity = df['Relative Humidity']
cloudCover = df['Cloud Cover']
Pressure = df['Sea Level Pressure']

# df2 pollutant columns

Date2 = df2['Date']
Time2 = df2['Time']
PM25 = df2['PM2.5']
PM10 = df2['PM10']
nitOx = df2['NO']
nitDiox = df2['NO2']
nitOxs = df2['NOx']
Ammonia = df2['NH3']
carbonMonox = df2['CO']
sulfDiox = df2['SO2']
Ozone = df2['O3']
Benzene = df2['Benzene']
Toluene = df2['Toluene']
Xylene = df2['Xylene']

# creating shifted ozone columns

# code to do so:
# print(Ozone.mean())

# for x in range(1, 25):
#     print('OzoneP%d = Ozone.shift(%d)' %(x, x))
#     print('OzoneP%d = OzoneP%d.replace(np.nan, 57.33)' %(x, x))

# for x in range(1, 25):
#     print("df3['O3P%d'] = OzoneP%d" %(x, x))

OzoneP1 = Ozone.shift(1)
OzoneP1 = OzoneP1.replace(np.nan, 57.33)
OzoneP2 = Ozone.shift(2)
OzoneP2 = OzoneP2.replace(np.nan, 57.33)
OzoneP3 = Ozone.shift(3)
OzoneP3 = OzoneP3.replace(np.nan, 57.33)
OzoneP4 = Ozone.shift(4)
OzoneP4 = OzoneP4.replace(np.nan, 57.33)
OzoneP5 = Ozone.shift(5)
OzoneP5 = OzoneP5.replace(np.nan, 57.33)
OzoneP6 = Ozone.shift(6)
OzoneP6 = OzoneP6.replace(np.nan, 57.33)
OzoneP7 = Ozone.shift(7)
OzoneP7 = OzoneP7.replace(np.nan, 57.33)
OzoneP8 = Ozone.shift(8)
OzoneP8 = OzoneP8.replace(np.nan, 57.33)
OzoneP9 = Ozone.shift(9)
OzoneP9 = OzoneP9.replace(np.nan, 57.33)
OzoneP10 = Ozone.shift(10)
OzoneP10 = OzoneP10.replace(np.nan, 57.33)
OzoneP11 = Ozone.shift(11)
OzoneP11 = OzoneP11.replace(np.nan, 57.33)
OzoneP12 = Ozone.shift(12)
OzoneP12 = OzoneP12.replace(np.nan, 57.33)
OzoneP13 = Ozone.shift(13)
OzoneP13 = OzoneP13.replace(np.nan, 57.33)
OzoneP14 = Ozone.shift(14)
OzoneP14 = OzoneP14.replace(np.nan, 57.33)
OzoneP15 = Ozone.shift(15)
OzoneP15 = OzoneP15.replace(np.nan, 57.33)
OzoneP16 = Ozone.shift(16)
OzoneP16 = OzoneP16.replace(np.nan, 57.33)
OzoneP17 = Ozone.shift(17)
OzoneP17 = OzoneP17.replace(np.nan, 57.33)
OzoneP18 = Ozone.shift(18)
OzoneP18 = OzoneP18.replace(np.nan, 57.33)
OzoneP19 = Ozone.shift(19)
OzoneP19 = OzoneP19.replace(np.nan, 57.33)
OzoneP20 = Ozone.shift(20)
OzoneP20 = OzoneP20.replace(np.nan, 57.33)
OzoneP21 = Ozone.shift(21)
OzoneP21 = OzoneP21.replace(np.nan, 57.33)
OzoneP22 = Ozone.shift(22)
OzoneP22 = OzoneP22.replace(np.nan, 57.33)
OzoneP23 = Ozone.shift(23)
OzoneP23 = OzoneP23.replace(np.nan, 57.33)
OzoneP24 = Ozone.shift(24)
OzoneP24 = OzoneP24.replace(np.nan, 57.33)

## was done to plot significance of models over time ^

# correlation with each input variable:

df2['Temp'] =df['Temperature']
df2['Humidity'] = df['Relative Humidity']
df2['Cloud Cover'] = df['Cloud Cover']
df2['O3P24'] = OzoneP24

vars = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3','CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Temp', 'Humidity','Cloud Cover']

corr_matrix = (df2.corr(method = 'pearson'))
corr_matrix = round(corr_matrix, 3)
print("correlation between O3 and other pollutants: ", len(df2))
print(corr_matrix['O3'].sort_values(ascending = False))

names = []
corrs = []

for x in range(len(vars)):
    currVarname = vars[x]
    currVar = df2[currVarname]
    
    names.append(currVarname)
    corrs.append((round(currVar.corr(OzoneP24), 3)))

print(names, corrs)

df4 = pd.DataFrame(list(zip(names, corrs)), 
               columns =['Variable Name', 'Correlation']) 
df4.sort_values('Correlation',inplace=True)


fig = plt.figure(figsize=(30, 16))
ax4 = df4.plot(kind='barh',y='Correlation',x='Variable Name',color='b', title='Correlation of Pollutants with Ozone Concentration in 24 Hours')
ax4.set_xlabel('Correlation Coefficient')

# generate plots ozone over 5 days (mon-fri) in the first week of the middle month of each season
Ozonethis = df2['O3']
nitDiox = df2['NO2']
# Ozonethis = df2['O3P24']
# Ozonethis = df2['O3']

dateCheck = 119
dateCheck2 = 262

OzoneP1 = Ozonethis[dateCheck:dateCheck2]
Ozonethis = Ozonethis[dateCheck:dateCheck2]
nitDiox = nitDiox[dateCheck:dateCheck2]

df.head()
print(Date2[dateCheck], Time2[dateCheck])
print(Date2[dateCheck2], Time2[dateCheck2])


# winter (jan) plot:

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
ax1.plot(Ozonethis.index, Ozonethis,'--', label = 'Ozone')
ax1.plot(nitDiox.index, nitDiox, label = 'Nitrogen Dioxide')
plt.xlabel('Hour (Index)')
plt.ylabel('Pollutant Concentration (µg/m³)')
plt.grid()
plt.title('Hourly Pollutant Levels from 1/6-1/11')
plt.legend(loc = 0, fancybox=True, framealpha=1, shadow=True, borderpad=1, prop={'size': 10})
plt.savefig('figs/xtra/Hourly Pollutant Levels 1.png')

# generate plots ozone over 5 days (mon-fri) in the first week of the middle month of each season

Ozone = df2['O3']
nitDiox = df2['NO2']

dateCheck = 2279
dateCheck2 = 2446

Ozone = Ozone[dateCheck:dateCheck2]
nitDiox = nitDiox[dateCheck:dateCheck2]

df.head()
print(Date2[dateCheck], Time2[dateCheck])
print(Date2[dateCheck2], Time2[dateCheck2])

# winter (jan) plot:

fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)
ax1.plot(Ozone.index, Ozone)
ax1.plot(nitDiox.index, nitDiox)
plt.xlabel('Date (Index)')
plt.ylabel('O3 Concentration)')
plt.title('Hourly Ozone Temperatures from 4/5-4/11')

## plotting average daily ozone concentration over a day for each season:

Ozone = df2['O3']

# all seasons - saved
avgO3Hour = []
currHour = []


for z in range(0, 24):
    tempCol = (Ozone[z::24])
    tempAvg = tempCol.mean()
#     print(z + 1, tempAvg)
    # currHour.append(z)
    avgO3Hour.append(z + 1)
    avgO3Hour.append(tempAvg)
    
# all
theHour = avgO3Hour[0::2]
print(theHour, len(theHour))

theAvg = avgO3Hour[1::2]
print(theAvg, len(theAvg))

# winter1
theHour1 = avgO3Hour1[0::2]
print(theHour1, len(theHour1))

theAvg1 = avgO3Hour1[1::2]
print(theAvg1, len(theAvg1))

# spring2
theHour2 = avgO3Hour2[0::2]
print(theHour2, len(theHour2))

theAvg2 = avgO3Hour2[1::2]
print(theAvg2, len(theAvg2))

# summre3
theHour3 = avgO3Hour3[0::2]
print(theHour3, len(theHour3))

theAvg3 = avgO3Hour3[1::2]
print(theAvg3, len(theAvg3))

# fall4
theHour4 = avgO3Hour4[0::2]
print(theHour4, len(theHour4))

theAvg4 = avgO3Hour4[1::2]
print(theAvg4, len(theAvg4))

theHour = list(np.roll(theHour, -5))
theAvg = list(np.roll(theAvg, -5))
theHour1 = list(np.roll(theHour1, -5))
theAvg1 = list(np.roll(theAvg1, -5))
theHour2 = list(np.roll(theHour2, -5))
theAvg2 = list(np.roll(theAvg2, -5))
theHour3 = list(np.roll(theHour3, -5))
theAvg3 = list(np.roll(theAvg3, -5))
theHour4 = list(np.roll(theHour4, -5))
theAvg4 = list(np.roll(theAvg4, -5))

xAx = range(len(theAvg))

fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(111)
ax1.plot(xAx, theAvg, 'k--', label='Average')
ax1.plot(xAx, theAvg4, 'gold', label='Fall')
ax1.plot(xAx, theAvg3, 'r', label='Summer')
ax1.plot(xAx, theAvg2, 'g', label='Spring')
ax1.plot(xAx, theAvg1, 'b', label='Winter')
plt.grid(b= True)
plt.legend(loc = 1, fancybox=True, framealpha=1, shadow=True, borderpad=1, prop={'size': 9})
# plt.vlines(7, 20, 40, color = 'r')
plt.xlabel('Hours After Sunrise (7:00 am)', size = 13)
plt.ylabel('Average O3 Concentration (µg/m³)', size = 13)
plt.title('Average Hourly O3 Concentration Over One Day - All Seasons + Average', size = 14)

# seaborn heatmap

import seaborn as sns
plt.figure(figsize = (17,8))
plt.title("Delhi Pollutant and Weather Correlations", size = 16)
sns.heatmap(df2.corr(), annot = True, linewidth = 1.0, cmap= 'YlGnBu')

# individual correlation plots

df5 = df2.sample(n=2500)
print(len(df5))

Date2 = df5['Date']
PM25 = df5['PM2.5']
PM10 = df5['PM10']
nitOx = df5['NO']
nitDiox = df5['NO2']
nitOxs = df5['NOx']
Ammonia = df5['NH3']
carbonMonox = df5['CO']
sulfDiox = df5['SO2']
Ozone = df5['O3']
Temp = df5['Temperature']

currPol = sulfDiox
polCorres = (round(Ozone.corr(currPol),4))
nowPol = ('Sulfur Dioxide')

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
plt.scatter(Ozone, currPol, s = 4)
# calc the trendline
m, b = np.polyfit(Ozone, currPol, 1)
plt.plot(Ozone, m*Ozone + b)
y = 1*x+0
plt.xlim([0, 200])
plt.ylim([0, 40])
plt.grid(b= True)

plt.title("Ozone vs %s - Correlation of %f" % (nowPol, polCorres), size = 16)
plt.xlabel('Ozone Concentration (µg/m³)', size = 13)
plt.ylabel('%s Concentration (µg/m³)' %(nowPol), size = 13)
plt.savefig('figs/vars/Ozone vs %s.png' % nowPol)

print(polCorres, type(polCorres))
