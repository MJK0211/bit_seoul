import numpy as np
import pandas as pd
import datetime as dt

day = np.array(['20110101', '20110202', '20110203', '20110204', '20110301', '20110505',
                     '20110510', '20110606','20110815', '20110911', '20110912', '20110913',
                     '20111003', '20111225'])
change_day = list()

for i in range(len(day)):
    change_day.append(dt.datetime.strptime(day[i], '%Y%m%d').strftime('%Y-%m-%d'))
    
# print(change_day)

df = pd.read_csv('./project/미국테스트.csv',
                  # parse_dates=['날짜'],
                  index_col=None,
                  header=0,
                  sep=',')  #(1200,12)

df = df.sort_values(['날짜'], ascending=['True'])
df = df.drop(['변동'], axis=1)

for i in range(len(df.index)):
    df.iloc[i,0] = df.iloc[i,0].replace("년","")
    df.iloc[i,0] = df.iloc[i,0].replace("월","")
    df.iloc[i,0] = df.iloc[i,0].replace("일","")
    df.iloc[i,0] = df.iloc[i,0].replace(" ","-")

    for j in range(len(df.iloc[i])-1):       
         df.iloc[i,j+1] = float(df.iloc[i,j+1].replace(",",""))

df = df.values 

# print(df.shape) #(260, 5)
# print(len(df))
# '''
find_index = list()
for i in range(len(df)):
    for j in range(len(change_day)):
        if df[i,0] == change_day[j]:
        #    print(df[i,0], i)
        #    print("ch : ", j)
        
            find_index.append(i)
        
            # dataset.drop('Name', axis=1, inplace=True)

print(find_index)
print(df.shape)
df = np.delete(df, find_index, axis=0)

print(df.shape)
# if change_day[1] == df.iloc[22,0]:
#     print('ok')
# print(change_day[1])
# print(df.iloc[22,0])
# df['날짜'] = pd.to_datetime(df['날짜'])
# df['날짜'] = df['날짜'] + dt.timedelta(days=+1)
# df['요일'] = df['날짜'].dt.day_name() #요일 설정
# df = df[(df['요일'] != 'Saturday') & (df['요일'] != 'Sunday')]
