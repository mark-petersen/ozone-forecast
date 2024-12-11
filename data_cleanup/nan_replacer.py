##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##
df = pd.read_csv('api_delhi_2015.csv')

print(df.columns)


# (all) weatherNames = ['Temperature','Dew Point', 'Relative Humidity', 'Heat Index', 'Wind Speed','Wind Gust', 'Wind Direction', 'Wind Chill', 'Precipitation','Precipitation Cover', 'Snow Depth', 'Visibility', 'Cloud Cover','Sea Level Pressure', 'Weather Type', 'Latitude', 'Longitude','Resolved Address', 'Name', 'Info', 'Conditions']

# actual ones being checked
weatherNames = ['Temperature','Relative Humidity', 'Cloud Cover', 'Sea Level Pressure']

for b in range(len(weatherNames)):
    currWea = df[(weatherNames[b])]
    currName = str(weatherNames[b])
    

    nanSum = currWea.isna().sum()
    nonNan = currWea.count()
    nanPercent = round((nonNan / len(currWea) * 100), 2)

    print(currName, "comp percentage: ", nanPercent)

df = pd.read_csv('api_delhi_2015.csv')

### fixing hourly data

fixedNum = 0
unfixedNum = 0

# make this into a loop for all variables first

weatherNames = ['Temperature','Relative Humidity', 'Cloud Cover', 'Sea Level Pressure']

# if now nan:
    # use prevnan 
    
# make prevWea the last non nan value
# make nextWea the soonest non nan value
    
print("starting nan checker")
    
for x in range(len(weatherNames)):
    currWea = df[weatherNames[x]]
    
    nanSum = currWea.isna().sum()
    nonNan = currWea.count()
    nanPercent = ((nonNan / len(currWea) * 100), 2)
    
    print("total number of nans:", nanSum)
    
    for b in range(len(df)):
        if b == 0:
            print("current variable: ", weatherNames[x])
            print("first index: ", b)
            pass
        
        if b + 1 == len(df):
            print("final index: ", b)
        
        if b > 2 and b + 1 != len(df):
            
            k = 1
            nowWea = currWea[b]
            prevWea = currWea[b - k]
            nextWea = currWea[b + k]
            
            if np.isnan(prevWea):
                for m in range(0, 200):
                    #k = 1
                    k += 1
                    prevWea = currWea[b - k]
                    if np.isnan(prevWea):
                        continue

                    if not np.isnan(prevWea):
                        break
                        
            if np.isnan(nextWea):
                for m in range(0, 200):
                    #k = 1
                    k += 1
                    nextWea = currWea[b - k]
                    if np.isnan(nextWea):
                        continue

                    if not np.isnan(nextWea):
                        break

            
            if np.isnan(nowWea) and not np.isnan(prevWea) and not np.isnan(nextWea):
                avgNan = (prevWea + nextWea) / 2
                df.loc[b, (weatherNames[x])] = avgNan
                fixedNum += 1

            
            else:
                unfixedNum += 1
                pass

        
    print("current variable: ", weatherNames[x])
    print("old total nans: ", nanSum)
    print("old completion percentage: ", nanPercent)
    
    nonNan = currWea.count()
    nanPercent = ((nonNan / len(currWea) * 100), 2)
    nanSum = currWea.isna().sum()
    
    print("new completion percentage: ", nanPercent)
    print("number of fixed values: ", fixedNum)
    print("number of unfixed values: ", unfixedNum)
    print("remaining number of nans: ", nanSum)
    print(currWea.head())

print("finished")
