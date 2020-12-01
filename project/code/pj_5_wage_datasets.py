import numpy as np
import pandas as pd

df = pd.read_csv('./project/data/csv/최저임금.csv',              
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',') 

df = df.sort_values(['연도'], ascending=['True'])

df = df.values
print(df)
np.save('./project/data/npy/wage.npy', arr=df)