import numpy as np
import pandas as pd
import datetime as dt

df = pd.read_csv('./data/project/미국테스트.csv',
                  # parse_dates=['날짜'],
                  index_col=None,
                  header=0,
                  sep=',')  #(1200,12)

df = df.sort_values(['날짜'], ascending=['True'])

for i in range(len(df.index)):
    df.iloc[i,0] = df.iloc[i,0].replace("년","")
    df.iloc[i,0] = df.iloc[i,0].replace("월","")
    df.iloc[i,0] = df.iloc[i,0].replace("일","")
    df.iloc[i,0] = df.iloc[i,0].replace(" ","-")

    if df.iloc[i,0] == '2011-08-25' :
        df_sub = i

print(df_sub)
df = df[df_sub-1:]
df = df.drop([0])

df['날짜'] = pd.to_datetime(df['날짜'])
df['날짜'] = df['날짜'] + dt.timedelta(days=+1)
df['요일'] = df['날짜'].dt.day_name() #요일 설정
df = df[(df['요일'] != 'Saturday') & (df['요일'] != 'Sunday')]


print(df.isnull().sum())
