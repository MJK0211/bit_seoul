from keras.models import Sequential
from keras.layers import Dense
import numpy as np 

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

x_predict = np.array([21,22,23,24,25])

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=1)

loss, accuracy = model.evaluate(x_test, y_test, batch_size=1)

print('loss: ', loss)
print('acc : ', accuracy)

y_predict = model.predict(x_predict)
print(y_predict)