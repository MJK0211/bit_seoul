import numpy as np
import pandas as pd
import datetime as dt

# data1 = [['2011-02-01',2,3], 
#          ['2011-02-03',7,8], 
#          ['2011-02-04', 5, 1], 
#          ['2011-02-05', 3, 4]]
# data2 = [['2011-02-01',222,3333], 
#          ['2011-02-04',777,888], 
#          ['2011-02-06', 555, 111], 
#          ['2011-02-07', 333, 444]]

# df1 = pd.DataFrame(data1,                
#                    columns=['날짜','종가','고가'])

# df2 = pd.DataFrame(data2,                
#                    columns=['날짜','종가','고가'])

df1 = pd.read_csv('./project/날씨테스트.csv',
                  # parse_dates=['날짜'],
                  index_col=None,
                  header=0,
                  encoding='cp949',
                  sep=',')  #(1200,12)

df1 = df1.sort_values(['날짜'], ascending=['True'])
df1['날짜'] = pd.to_datetime(df1['날짜'])
# df1['날짜'] = pd.to_datetime(df1['날짜'])

# df1['요일'] = df1['날짜'].dt.day_name() #요일 설정

# df1 = df1.drop(['지점'], axis=1) #지점 삭제
# df1 = df1[(df1['요일'] != 'Saturday') & (df1['요일'] != 'Sunday')]
# print(df1[34:])

df2 = pd.read_csv('./project/미국테스트.csv',
                  index_col=None,
                  header=0,           
                  sep=',')  #(1200,12)

df2 = df2.sort_values(['날짜'], ascending=['True'])

for i in range(len(df2.index)):
    df2.iloc[i,0] = df2.iloc[i,0].replace("년","")
    df2.iloc[i,0] = df2.iloc[i,0].replace("월","")
    df2.iloc[i,0] = df2.iloc[i,0].replace("일","")
    df2.iloc[i,0] = df2.iloc[i,0].replace(" ","")

df2['날짜'] = pd.to_datetime(df2['날짜'])
df2['요일'] = df2['날짜'].dt.day_name() #요일 설정
df2 = df2[(df2['요일'] != 'Saturday') & (df2['요일'] != 'Sunday')]

# df2_list = list()
# print(len(df1.index))
# print(len(df2.index))
# print(len(df1.iloc[0]))
# print(len(df2.iloc[0]))

df_list = list()

for i in range(len(df1.index)):
    # print(df1.iloc[i,0])
    for j in range(len(df2.index)):
        # print(df2.iloc[j,0])    
        if df1.iloc[i,0] == df2.iloc[j,0]:
            for k in range(len(df1.iloc[i])):
                df_list.append(df1.iloc[i,k])
            # df_list = np.array(df_list)
              
print(df_list[0]) 
    # for j in range(len(df2.iloc[i])):
    #     # if df1.iloc[i,0] == df2_list[i]:
    #     print(i)  
           
# print(df1)
# df = pd.read_csv('./data/project/미국테스트.csv',
#                   # parse_dates=['날짜'],
#                   index_col=None,
#                   header=0,
#                   sep=',')  #(1200,12)

# df = df.sort_values(['날짜'], ascending=['True'])
