#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y = np.array((range(101,201), range(311,411), range(100)))

print(x)
print(x.shape) #(3,100)

# 과제(100,3)으로 바꿔보기
# x = x.T - [[1,711,0], [2,712,1] ...] 형태
# print(x.shape) 

x=x.reshape(100,3)

print(x)  # [[1,2,3], [4,5,6] ...] 형태
print(x.shape)
