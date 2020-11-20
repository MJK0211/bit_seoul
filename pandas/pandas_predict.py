import numpy as np
import pandas as pd


df1 = pd.read_csv('./data/csv/비트컴퓨터 1120.csv',
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',')  #(1200,12)
bit = df1.iloc[:627, [0,1,2,3,7]]              
bit = bit.sort_values(['일자'], ascending=['True'])

for i in range(len(bit.index)):
    for j in range(len(bit.iloc[i])):
        bit.iloc[i,j] = int(bit.iloc[i,j].replace(',',''))
#print(bit.shape) #(1200, 5)
print(bit)
bit = bit.values
'''
print(bit)

df2 = pd.read_csv('./data/csv/삼성전자 1120.csv',
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',') #(660,12)
samsung = df2.iloc[:, [0,1,2,3,7,8]]
samsung = samsung.sort_values(['일자'], ascending=['True'])

#print(samsung)
#                    시가         고가         저가         종가         거래량     금액(백만)
# 일자
# 2018/03/19  2,531,000  2,567,000  2,522,000  2,537,000     164,377    417,387
# 2018/03/20  2,535,000  2,560,000  2,505,000  2,560,000     163,865    414,791
# 2018/03/21  2,589,000  2,589,000  2,553,000  2,553,000     178,104    456,127
print(type(samsung)


for k in range(len(samsung.index)):
    for l in range(len(samsung.iloc[k])):
           samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',''))
print(samsung)

#######################################################
# bit.shape - 시가 고가 저가 종가 거래량 #(1200, 5)
# samsung.shape - 시가 고가 저가 종가 거래량 금액(백만) #(660, 6)
#######################################################
'''