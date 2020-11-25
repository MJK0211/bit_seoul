# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# load_boston dnn 과 loss / acc를 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.datasets import load_boston #dataset인 load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터
boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target , train_size=0.8, random_state=42, shuffle=True)

print(x_train.shape) #(404, 13)
print(x_test.shape) #(102, 13)

x_train_test = np.append(x_train, x_test, axis=0)
print(x_train_test.shape) #(506, 13)

# pca = PCA()
# pca.fit(x_train_test)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# n_components1 = np.argmax(cumsum >=0.99) + 1 #3
# n_components2 = np.argmax(cumsum >=1.00) #13
# print(cumsum)
# print(n_components1)
# print(n_components2) 


pca = PCA(n_components=3)
x2d = pca.fit_transform((x_train_test))
x_train = x2d[:404]
x_test = x2d[404:496]
x_pred = x2d[496:]

y_real = y_test[92:]
y_test = y_test[:92]


#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(3,))) 
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
# loss :  91.15850067138672
# y_real :  [23.6 32.4 13.6 22.8 16.1 20.  17.8 14.  19.6 16.8]
# y_pred :  [24.133478 29.340923 33.484577 22.216171 28.625301  9.904315 22.044025
#  13.546109 28.957869 19.76713 ]