#1. 데이터
import numpy as np
x = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y = np.array((range(101,201), range(311,411), range(100)))

print(x)
x = x.T
print(x)
print(x.shape)

# 과제(100,3)

