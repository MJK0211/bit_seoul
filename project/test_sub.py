import numpy as np
import pandas as pd
import datetime as dt

data1 = [['2011-02-01',2,3], 
         ['2011-02-03',7,8], 
         ['2011-02-04', 5, 1], 
         ['2011-02-05', 3, 4]]
data2 = [['2011-02-01',222,3333], 
         ['2011-02-04',777,888], 
         ['2011-02-06', 555, 111], 
         ['2011-02-07', 333, 444]]

df1 = pd.DataFrame(data1,                
                   columns=['날짜','종가','고가'])

df2 = pd.DataFrame(data2,                
                   columns=['날짜','종가','고가'])

df2_list = list()
df1_list = list()
for i in range(len(df2.index)):
    df2_list.append(df2.iloc[i,0])    
            
    for j in range(len(df2.iloc[i])):
        # if df1.iloc[i,0] == df2_list[i]:
        print(i)  
           
# print(df1)
