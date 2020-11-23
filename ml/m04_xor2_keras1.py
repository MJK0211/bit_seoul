import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측

y_pred = model.predict(x_data)
print(x_data, "의 예측결과 : \n", y_pred)

loss = model.evaluate(x_data, y_data)
print("loss : ", loss)

acc_score = accuracy_score(y_data, y_pred.round())
print("acc_score : ", acc_score)

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  
# [[0.47840393]
#  [0.7053647 ]
#  [0.48446956]
#  [0.71038884]]
# 1/1 [==============================] - 0s 1ms/step - loss: 0.7410 - acc: 0.5000
# loss :  [0.7409546375274658, 0.5]
# acc_score :  0.5