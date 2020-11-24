import numpy as np
import pandas as pd

df1 = pd.read_csv('./data/project/최저임금.csv',
                       header=0,
                       index_col=0,
                       sep=',',
                       encoding='cp949')
# print(df1)

df1 = pd.read_csv('./data/project/날씨.csv',
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',')  #(1200,12)
print(df1)