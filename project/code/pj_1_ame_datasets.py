import numpy as np
import pandas as pd
import datetime as dt

holiday = np.load('./project/data/npy/holiday.npy')
holiday_k_stock = np.load('./project/data/npy/holiday_k_stock.npy')
check_holiday = np.load('./project/data/npy/check_holiday.npy')

holiday_all = np.hstack((holiday,holiday_k_stock,check_holiday))
holiday_all = np.unique(holiday_all) 

df = pd.read_csv('./project/data/csv/미국환율.csv',
                  # parse_dates=['날짜'],
                  index_col=None,
                  header=0,
                  sep=',') 


df = df.drop(['변동 %'], axis=1)
df = df.sort_values(['날짜'], ascending=['True'])

for i in range(len(df.index)):
    df.iloc[i,0] = df.iloc[i,0].replace("년","")
    df.iloc[i,0] = df.iloc[i,0].replace("월","")
    df.iloc[i,0] = df.iloc[i,0].replace("일","")
    df.iloc[i,0] = df.iloc[i,0].replace(" ","-")

    for j in range(len(df.iloc[i])-1):       
         df.iloc[i,j+1] = float(df.iloc[i,j+1].replace(",",""))

    if df.iloc[i,0] == '2011-08-25' :
        df_sub = i
     
df = df[df_sub:]
# df = df.drop([0])
df['날짜'] = pd.to_datetime(df['날짜'])
df['요일'] = df['날짜'].dt.day_name() #요일 설정
df = df[(df['요일'] != 'Saturday') & (df['요일'] != 'Sunday')] #토요일, 일요일 삭제
df = df.drop(['요일'], axis=1) #요일 삭제
# df['날짜'] = df['날짜'] + dt.timedelta(days=+1)
df['날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')

df = df.values

find_index = list()
find2_index = list()

for i in range(len(df)):
    for j in range(len(holiday_all)):
        if df[i,0] == holiday_all[j]:
            find_index.append(i)
    
print(find_index)

print(df.shape)
df = np.delete(df, find_index, axis=0)
print(df.shape) #(2275,4)

np.save('./project/data/npy/america.npy', arr=df)
