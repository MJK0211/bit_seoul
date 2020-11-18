#load_breast_cancer - DNN - 이진분류(유방암) 걸렸으면1, 아니면2

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_breast_cancer #load_breast_cancer 이진분류(유방암) 걸렸으면1, 아니면2

dataset = load_breast_cancer()
x = dataset.data #(569, 30)
y = dataset.target #(569,)

print(x[0])
print(y[0])

print(x.shape)
print(y.shape)

x = x[10:] 
y = y[10:]
x_pred = x[:10]
y_real = y[:10]

print(x.shape) #(559, 30)
print(x_pred.shape) #(10, 30)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_minmax = scaler.transform(x) 
x_pred_minmax = scaler.transform(x_pred)
x_minmax = x_minmax.reshape(559,15,2,1)
x_pred_minmax = x_pred.reshape(10,15,2,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_minmax, y, train_size=0.8) 

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(15,2,1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(20, (2,2), padding='same')) 
# model.add(Dropout(0.2))
# model.add(Conv2D(30, (2,2), padding='same'))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=1)) 
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min') 

model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_pred_minmax)
y_predict = np.round(y_predict,1)
y_predict = y_predict.reshape(10,)
print("y_real : ", y_real)
print("y_predict : \n", y_predict)

