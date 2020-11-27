import numpy as np
import pandas as pd
import datetime as dt

df = pd.read_csv('./data/project/날씨.csv',
                  # parse_dates=['날짜'],
                  index_col=None,
                  header=0,
                  encoding='cp949',
                  sep=',')  #(1200,12)

df['날짜'] = pd.to_datetime(df['날짜'])

df['요일'] = df['날짜'].dt.day_name() #요일 설정

df = df.drop(['지점'], axis=1) #지점 삭제
df = df[(df['요일'] != 'Saturday') & (df['요일'] != 'Sunday')]

# ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
# python은 and나 or 가 true values를 요구하지만 이런 표현이 모호할 수 있다

print(df[34:])


