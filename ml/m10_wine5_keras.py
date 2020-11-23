import numpy as np
import pandas as pd

wine = pd.read_csv('./data/csv/winequality-white.csv',
                 header = 0,
                 index_col=None,
                 sep=';')

y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)

#3~9 나누기

newlist = []
for i in list(y):
    if i <= 4:
        newlist +=[0]
    elif i<=7:
        newlist +=[1]
    else :
        newlist +=[2] #데이터 조작이 아닌가? -> 전처리일수 있다, 와인의 품질을 판단하는 데이터셋
                      #와인의 등급 3~9등급을 맞추는 데이터셋이지만, but 0,1,2단계로 라벨링을 다시 해준 것!
y = newlist
y = np.asarray(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8) #shuffle로 섞을 경우, random난수로 섞는다

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train)
x_test_minmax = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(11,)))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train_minmax, y_train, batch_size=1, epochs=10)

#4. 평가, 예측
loss, acc = model.evaluate(x_test_minmax, y_test)
print("loss : ", loss)
print("acc : ", acc)
y_pred = model.predict(x_test_minmax)
print(np.argmax(y_test[20:30], axis=1), "의 예측결과 : \n", np.argmax(y_pred[20:30], axis=1))


# score = model.score(x_test_minmax, y_test)
# print("score : ", score)

# from sklearn.metrics import accuracy_score
# acc_score = accuracy_score(y_test, y_pred)
# print("acc_score : ", acc_score)

# 결과값
# loss :  0.27715742588043213
# acc :  0.9265305995941162
# [1 0 1 1 1 1 1 1 1 1] 의 예측결과 :
#  [1 1 1 1 1 1 1 1 1 1]