import numpy as np
import pandas as pd


datasets = pd.read_csv('./data/csv/iris_ys.csv', 
                      header=0, #header는 데이터포함 X
                      index_col=0, #index 라인 적용
                      sep=',') # Default는 ',' 이다
print(datasets)

#      sepal_length  sepal_width  petal_length  petal_width  species
# 2             4.9          3.0           1.4          0.2        0
# 3             4.7          3.2           1.3          0.2        0
# 4             4.6          3.1           1.5          0.2        0
# 5             5.0          3.6           1.4          0.2        0
# ..            ...          ...           ...          ...      ...
# 146           6.7          3.0           5.2          2.3        2
# 147           6.3          2.5           5.0          1.9        2
# 148           6.5          3.0           5.2          2.0        2
# 149           6.2          3.4           5.4          2.3        2
# 150           5.9          3.0           5.1          1.8        2

print(datasets.shape) #(150, 5)

# datasets = pd.read_csv('./data/csv/iris_ys.csv', 
#                       header=None, #header=None 하면 header에 새로운 인덱스 생성
#                       index_col=None, #index_col=None 하면 index도 데이터로 들어간다
#                       sep=',')

#          0             1            2             3            4        5
# 0      NaN  sepal_length  sepal_width  petal_length  petal_width  species
# 1      1.0           5.1          3.5           1.4          0.2        0
# 2      2.0           4.9            3           1.4          0.2        0
# 3      3.0           4.7          3.2           1.3          0.2        0
# 4      4.0           4.6          3.1           1.5          0.2        0
# ..     ...           ...          ...           ...          ...      ...
# 146  146.0           6.7            3           5.2          2.3        2
# 147  147.0           6.3          2.5             5          1.9        2
# 148  148.0           6.5            3           5.2            2        2
# 149  149.0           6.2          3.4           5.4          2.3        2
# 150  150.0           5.9            3           5.1          1.8        2

# print(datasets.shape) #(151, 6)

print(datasets.head()) #위에서부터 5개

#  sepal_length  sepal_width  petal_length  petal_width  species
# 1           5.1          3.5           1.4          0.2        0
# 2           4.9          3.0           1.4          0.2        0
# 3           4.7          3.2           1.3          0.2        0
# 4           4.6          3.1           1.5          0.2        0
# 5           5.0          3.6           1.4          0.2        0

print(datasets.tail()) #아래에서부터 5개

#  sepal_length  sepal_width  petal_length  petal_width  species
# 146           6.7          3.0           5.2          2.3        2
# 147           6.3          2.5           5.0          1.9        2
# 148           6.5          3.0           5.2          2.0        2
# 149           6.2          3.4           5.4          2.3        2
# 150           5.9          3.0           5.1          1.8        2

print(type(datasets)) #<class 'pandas.core.frame.DataFrame'>

aaa = datasets.to_numpy() #1. pandas datasets를 numpy형태로 변환
#aaa = datasets.values    #2. pandas datasets를 numpy형태로 변환
print(type(aaa)) # <class 'numpy.ndarray'>
print(aaa.shape) # (150, 5)
print(aaa)

np.save('./data/iris_ys_pd.npy', arr=aaa) #pandas에서 numpy로 변환된 iris_ys_pd.npy 저장 