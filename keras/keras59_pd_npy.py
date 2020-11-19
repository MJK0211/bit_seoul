#실습
#iris_ys2.csv 파일을 넘파이로 불러오기
#불러온 데이터를 판다스로 저장
#모델 완성
import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris_ys2.csv', 
                      header=None, 
                      index_col=None, 
                      sep=',') 

aaa = datasets.to_numpy()
print(type(aaa)) # <class 'numpy.ndarray'>
print(aaa.shape) # (150, 5)

x = aaa[0:, :4]
y = aaa[0:, 4:]

x = pd.DataFrame(x)
y = pd.DataFrame(y)

'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.8, shuffle=False) 

x_pred = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_pred)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(4, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax')) 

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=5) 

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train_minmax, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
result = model.evaluate(x_test_minmax, y_test, batch_size=1)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_pred_minmax)
y_pred = np.argmax(y_predict, axis=1) 
print("y_real : \n", y_real)
print("y_pred : ", y_pred)


# bbb = pd.DataFrame(aaa)
# print(type(bbb)) #<class 'pandas.core.frame.DataFrame'>
# print(bbb.shape) #(150, 5)
# x = bbb.iloc[0:, :4]
# y = bbb.iloc[0:, 4:]
# print(x)
# print(y)
'''