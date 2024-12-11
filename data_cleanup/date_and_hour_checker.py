##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##
df = pd.read_csv('api_delhi_2015.csv')
Date = df['Date']
Time = df['Time']
#City = df['Address']
# Temp = df['Temperature']

#day = int(Date.values[20].split('/')[0])
print(day)

##### check whether the day is +1 the previous day, and if not print out the exceptions

nonConDays = ()

for x in range(len(df)):
    
    if x == 0:
        pass
    
    if (x + 1 < len(df)):
        day = int(Date.values[x].split('/')[0])
        prevDay = int(Date.values[x-1].split('/')[0])

        if day == prevDay:
            pass
            
        else:
            if day == prevDay + 1:
                pass
            
            if not day == prevDay + 1:
                
                print("it's the end of the month:", Date[x])
                    #pass
                
            else:
                print("nonconsecutive between ", prevDay, "and ", day, x, Date[x])
    else:
        print("final index: ", x)

##### check if every hour is one plus the previous one, if not print the exceptions

for i in Date.values[:50]:
    if type(i) != str:
        print(i)
    else:
        pass

dayCheck = df[::24]
newTime = dayCheck['Time']
fineCheck = 0
otherCounter = 0

for b in range(len(df)):
    if b == 0:
        otherCounter += 1
        pass
    
    elif (b+1 < len(df)):
        Hour = int(Time.values[b].split(':')[0])
        prevHour = int(Time.values[b-1].split(':')[0])
        
        if Hour == 0:
            if prevHour == 23:
                otherCounter += 1
                pass
            if prevHour != 23:
                print("nonconsecutive from 0: ", prevHour, Hour, Time[b], df.loc[b, 'Date'])
        
        else:
        
            if Hour == prevHour + 1:
                fineCheck += 1
                pass

            if Hour == prevHour:
                print("error- similar hours", prevHour, Hour)
                pass
            
            if Hour != prevHour + 1:
                print("nonconsecutive hours: ", prevHour, Hour, Time[b], df.loc[b, 'Date'])
        
    if b + 1 == len(df):
        print("final index", b)
        otherCounter += 1

totalCount = fineCheck + otherCounter
print("total len: ", totalCount, fineCheck, otherCounter)
print("total len should be 8760")
