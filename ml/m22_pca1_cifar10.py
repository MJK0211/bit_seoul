# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1이상
# cifar dnn 과 loss / acc를 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar10 #dataset인 cifar10
from sklearn.decomposition import PCA

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_test = np.append(x_train, x_test, axis=0)
print(x_train_test.shape) #(60000, 32, 32, 3)

x_train_test = x_train_test.reshape(x_train_test.shape[0],
                                    x_train_test.shape[1]*x_train_test.shape[2]*x_train_test.shape[3])


# pca = PCA()
# pca.fit(x_train_test)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)
# n_components1 = np.argmax(cumsum >=0.95) + 1 #217
# n_components2 = np.argmax(cumsum >=1) + 1 #3072
# print(n_components1)
# print(n_components2)

pca = PCA(n_components=3072)
# pca = PCA(n_components2)
x2d = pca.fit_transform((x_train_test))

x_train = x2d[:50000]
x_test = x2d[50000:59990]
x_pred = x2d[59990:]

y_real = y_test[9990:]
y_test = y_test[:9990]

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(3072,))) 
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu')) 
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # mse - 'mean_squared_error' 가능

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_pred) # np.argmax로 변환하지 않아도 같은 값으로 출력이 가능하다
y_pred = np.argmax(y_pred, axis=1)
print("y_real : ", y_real.reshape(10,))
print("y_pred : ", y_pred)

# 결과값
# loss :  2.7207653522491455
# acc :  0.40560561418533325
# y_real :  [7 0 3 5 3 8 3 5 1 7]
# y_pred :  [4 1 5 5 5 3 6 4 4 7]