import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100)

data = randn(5,4)

# print(data.shape) #(5, 4)

df = pd.DataFrame(data, 
                  index='A B C D E'.split(), #index
                  columns='가 나 다 라'.split()) #header

# print(df)
#           가         나         다         라
# A -1.749765  0.342680  1.153036 -0.252436
# B  0.981321  0.514219  0.221180 -1.070043
# C -0.189496  0.255001 -0.458027  0.435163
# D -0.583595  0.816847  0.672721 -0.104411
# E -0.531280  1.029733 -0.438136 -1.118318

data2 = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]

df2 = pd.DataFrame(data2, 
                   index=['A','B','C','D','E'], 
                   columns=['가','나','다','라'])

# print(df2)
#  가   나   다   라
# A   1   2   3   4
# B   5   6   7   8
# C   9  10  11  12
# D  13  14  15  16
# E  17  18  19  20

df3 = pd.DataFrame(np.array([[1,2,3], [4,5,6]]))
# print(df3) -> index, header 자동으로 생성
#    0  1  2 
# 0  1  2  3
# 1  4  5  6

# 컬럼
# print("df2['나'] : \n", df2['나'])
# df2['나'] : 
#  A     2
# B     6
# C    10
# D    14
# E    18

# print("df2['나', '라'] : \n", df2[['나','라']])
# df2['나', '라'] : 
#      나   라
# A   2   4
# B   6   8
# C  10  12
# D  14  16
# E  18  20

# print("df2[0] : \n", df2[0]) #에러

# print("df2.loc['나'] : \n", df2.loc['나']) #에러 loc 행에서만 사용가능! A, B, C, D, E

# print("df2.iloc[:, 2] : \n", df2.iloc[:, 2]) 
# df2.iloc[:, 2] : 
#  A     3
# B     7
# C    11
# D    15
# E    19

# print("df2[:, 2] : \n", df2[:, 2]) #에러

# print("df2.loc['A'] : \n", df2.loc['A'])
# df2.loc['A'] : 
#  가    1
# 나    2
# 다    3
# 라    4

#print("df2.loc[['A', 'C']] : \n", df2.loc[['A', 'C']])
# df2.loc[['A', 'C']] : 
#     가   나   다   라
# A  1   2   3   4
# C  9  10  11  12

# print("df2.iloc[0] : \n", df2.iloc[0])
# df2.iloc[0] : 
#  가    1
# 나    2
# 다    3
# 라    4

#print("df2.iloc[0,2] : \n", df2.iloc[[0,2]])
# df2.iloc[0,2] : 
#     가   나   다   라
# A  1   2   3   4
# C  9  10  11  12

#행렬
# print("df2.loc[['A','B'],['나','다']] : \n", df2.loc[['A','B'],['나','다']])
# df2.loc[['A','B'],['나','다']] : 
#     나  다
# A  2  3
# B  6  7

# print("df2.loc['E','다'] : \n", df2.loc['E','다'])
# df2.loc['E','다'] : 
#  19

# print("df2.iloc[4,2] : \n", df2.iloc[4,2])
# df2.iloc[4,2] : 
#  19

# print("df2.iloc[4][2] : \n", df2.iloc[4][2])
# df2.iloc[4][2] : 
#  19