#Fashion_Mnist show plt

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import fashion_mnist #dataset인 fashion_mnist 추가
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape) #(60000, 28, 28), (10000, 28, 28) train 60000개, test 10000개
print(y_train.shape, y_test.shape) #(60000,) (10000,) train 60000개, test 10000개

# 0 티셔츠/탑
# 1 바지
# 2 풀오버(스웨터의 일종)
# 3 드레스
# 4 코트
# 5 샌들
# 6 셔츠
# 7 스니커즈
# 8 가방
# 9 앵클 부츠

plt.imshow(x_train[200], 'gray')
plt.show()
