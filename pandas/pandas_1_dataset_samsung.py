import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 

#1. 데이터
######################################비트 2018/05/04 ~ 2020/11/19
df1 = pd.read_csv('./data/csv/비트컴퓨터 1120.csv',
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',')  #(1200,12)
                  
bit = df1.sort_values(['일자'], ascending=['True'])
bit = bit.iloc[573:1199, [0,1,2,3,7]]               

for i in range(len(bit.index)):
    for j in range(len(bit.iloc[i])):
        bit.iloc[i,j] = int(bit.iloc[i,j].replace(',',''))

#print(bit.shape) #(626, 5)
bit = bit.to_numpy()
#np.save('./data/npy/bit.npy', arr=bit)

######################################삼성 2018/05/04 ~ 2020/11/19
df2 = pd.read_csv('./data/csv/삼성전자 1120.csv',   
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',') #(660,12)
samsung = df2.sort_values(['일자'], ascending=['True'])
samsung = samsung.iloc[33:659, [0,1,2,3,7,8]]

for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',''))

#print(samsung.shape) #(626, 6)
samsung = samsung.to_numpy()
#np.save('./data/npy/samsung.npy', arr=samsung)
#####################################코스닥 2018/05/04 ~ 2020/11/19 # 0,3,7,8
df3 = pd.read_csv('./data/csv/코스닥.csv',   
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',') 
kosdaq = df3.sort_values(['일자'], ascending=['True'])
#print(kosdaq.shape) #(880, 14)
kosdaq = kosdaq.iloc[253:879, [0,1,2,3]]
for i in range(len(kosdaq.index)):
    for j in range(len(kosdaq.iloc[i])):
        kosdaq.iloc[i,j] = float(kosdaq.iloc[i,j])
#print(kosdaq.shape) #(626, 4)
#print(kosdaq)
kosdaq = kosdaq.to_numpy()
np.save('./data/npy/kosdaq.npy', arr=kosdaq)
#####################################금현물 2018/05/04 ~ 2020/11/19 #0,3,7
df4 = pd.read_csv('./data/csv/금현물.csv',   
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',')
gold = df4.sort_values(['일자'], ascending=['True'])
#print(gold.shape) #(810, 12)
gold = gold.iloc[183:809, [0,3,7]]

for i in range(len(gold.index)):
    for j in range(len(gold.iloc[i])):
        gold.iloc[i,j] = int(gold.iloc[i,j].replace(',',''))
#print(gold) #(626, 3)
gold = gold.to_numpy()
np.save('./data/npy/gold.npy', arr=gold)