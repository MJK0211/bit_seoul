import numpy as np
import pandas as pd

df1 = pd.read_csv('./data/csv/kospi200.csv',
                  index_col=0,
                  header=0,
                  encoding='UTF-8',
                  sep=',')
print(df1)
print(df1.shape) #(426, 5)
print(type(df1)) #<class 'pandas.core.frame.DataFrame'>

df2 = pd.read_csv('./data/csv/samsung.csv',
                  index_col=0,
                  header=0,
                  encoding='UTF-8',
                  sep=',')

print(df2)
print(df2.shape) #(426, 5)
print(type(df2)) #<class 'pandas.core.frame.DataFrame'>

# KOSPI200 거래량 - 거래량 str -> int 변경
for i in range(len(df1.index)): 
    df1.iloc[i,4] = int(df1.iloc[i,4].replace(',',''))
  
# 삼성전자의 모든 데이터 - 모든 str -> int 변경
for i in range(len(df2.index)):
    for j in range(len(df2.iloc[i])):
        df2.iloc[i,j] = int(df2.iloc[i,j].replace(',',''))

df1 = df1.values # <class 'numpy.ndarray'>, (426,5)
df2 = df2.values # <class 'numpy.ndarray'>, (426,5)

np.save('./data/npy/kospi200.npy', arr=df1)
np.save('./data/npy/samsung.npy', arr=df2)

# samsung = samsung.sort_values(['일자'], ascending=['True'])