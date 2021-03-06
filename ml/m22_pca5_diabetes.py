# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# load_diabetes dnn 과 loss / acc를 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.datasets import load_diabetes #dataset인 load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터
diabetes = load_diabetes()

x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target , train_size=0.8, random_state=42, shuffle=True)

print(x_train.shape) #(353, 10)
print(x_test.shape) #(89, 10)

x_train_test = np.append(x_train, x_test, axis=0)
print(x_train_test.shape) #(442, 10)

# pca = PCA()
# pca.fit(x_train_test)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# n_components1 = np.argmax(cumsum >=0.95) + 1 #8
# n_components2 = np.argmax(cumsum >=1) #9
# print(cumsum)
# print(n_components1)
# print(n_components2) 

pca = PCA(n_components=8)
x2d = pca.fit_transform((x_train_test))
x_train = x2d[:353]
x_test = x2d[353:432]
x_pred = x2d[432:]

y_real = y_test[79:]
y_test = y_test[:79]


#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(8,))) 
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu')) 
model.add(Dense(1)) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)

y_pred = model.predict(x_pred) # np.argmax로 변환하지 않아도 같은 값으로 출력이 가능하다
print("y_real : ", y_real)
print("y_pred : ", y_pred.reshape(10,))

# 결과값
# loss :  3178.29541015625
# y_real :  [233.  68. 190.  96.  72. 153.  98.  37.  63. 184.]
# y_pred :  [212.27065  109.97298  139.92918   52.68962   48.360085 113.97997
#   83.23114   83.132744  65.541504 163.65558 