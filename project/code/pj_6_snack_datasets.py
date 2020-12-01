import numpy as np
import pandas as pd

df = pd.read_csv('./project/data/csv/새우깡.csv',                
                  index_col=0,
                  header=0,
                  encoding='cp949',
                  sep=',') 

df = df.sort_values(['연도'], ascending=['True'])

# print(df.shape)
print(df)

df = df.values
np.save('./project/data/npy/snack.npy', arr=df)
