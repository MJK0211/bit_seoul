#load_breast_cancer - DNN - 이진분류(유방암) 걸렸으면1, 아니면2

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from sklearn.datasets import load_breast_cancer #load_breast_cancer 이진분류(유방암) 걸렸으면1, 아니면2

# dataset = load_breast_cancer()
# x = dataset.data #(569, 30)
# y = dataset.target #(569,)

# print(x.shape) #(559, 30)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.8) 

x_train = np.load('./data/cancer_x_train.npy')
x_test = np.load('./data/cancer_x_test.npy')
y_train = np.load('./data/cancer_y_train.npy')
y_test = np.load('./data/cancer_y_test.npy')

x_predict = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_predict)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_shape=(30,))) 
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto') 

model.fit(x_train_minmax, y_train, epochs=1000, batch_size=32, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_pred_minmax)
y_predict = np.round(y_predict.reshape(10,),0)
print("y_real : ", y_real)
print("y_predict : ", y_predict)

# 결과값
# loss :  0.036799535155296326
# acc :  0.9807692170143127
# y_real :  [0 1 1 1 1 1 1 1 0 1]
# y_predict :
#  [0. 1. 1. 1. 1. 1. 1. 1. 0. 1.]