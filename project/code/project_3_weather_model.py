from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import numpy as np

#1. 데이터

weather = np.load('./project/data/npy/weather.npy', allow_pickle=True) #(2275,4)
weather = np.delete(weather, 0, axis=1).astype('float32')

wage_result = np.load('./project/data/npy/wage_result.npy', allow_pickle=True) #(2275,)

#x데이터
x_train, x_test = train_test_split(weather, train_size=0.8, random_state=66, shuffle=True)

#y데이터
y_train, y_test = train_test_split(wage_result, train_size=0.8, random_state=66, shuffle=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성 
model = Sequential()
model.add(Dense(200, input_shape=(3,))) 
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu')) 
model.add(Dense(1)) 
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
hist = model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)

y_pred = model.predict(x_test[-1:])
print("y_pred : ", y_pred)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #단위 무엇인지 찾아보기
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()
