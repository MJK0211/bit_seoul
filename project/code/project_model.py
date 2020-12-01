
#다한 사람은 모델을 완성해서 결과 주석으로 적어놓을 것
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import numpy as np

#1. 데이터
nc = np.load('./project/data/npy/nc.npy', allow_pickle=True) #(2275,4)
am = np.load('./project/data/npy/america.npy', allow_pickle=True) #(2275,4)
weather = np.load('./project/data/npy/weather.npy', allow_pickle=True) #(2275,3)
snack_result = np.load('./project/data/npy/snack_result.npy', allow_pickle=True).astype('float32') #(2275,)
wage_result = np.load('./project/data/npy/wage_result.npy', allow_pickle=True).astype('float32') #(2275,)

nc = np.delete(nc, 0, axis=1).astype('float32')
am = np.delete(am, 0, axis=1).astype('float32')
weather = np.delete(weather, 0, axis=1).astype('float32')

snack_result = snack_result.reshape(2275,1)

nc_pred = nc[2274:]
nc = nc[:2274]
am_pred = am[2274:]
am = am[:2274]
weather_pred = weather[2274:]
weather = weather[:2274]
snack_pred = snack_result[2274:]
snack_result = snack_result[:2274]

y_real = wage_result[2274:]
y = wage_result[:2274]

# wage_result = wage_result.reshape(2275,1)

#x데이터
nc_train, nc_test, am_train, am_test = train_test_split(nc, am , train_size=0.8, random_state=66, shuffle=True)
weather_train, weather_test, snack_train, snack_test = train_test_split(weather, snack_result , train_size=0.8, random_state=66, shuffle=True)

#y데이터
y_train, y_test = train_test_split(y, train_size=0.8, random_state=66, shuffle=True)

#데이터 전처리
from sklearn.preprocessing import MinMaxScaler

#1.1 nc데이터
scaler1 = MinMaxScaler()
scaler1.fit(nc_train)
nc_train = scaler1.transform(nc_train)
nc_test = scaler1.transform(nc_test)
nc_pred = scaler1.transform(nc_pred)

#1.2 am데이터
scaler2 = MinMaxScaler()
scaler2.fit(am_train)
am_train = scaler2.transform(am_train)
am_test = scaler2.transform(am_test)
am_pred = scaler2.transform(am_pred)

#1.3 weather데이터
scaler3 = MinMaxScaler()
scaler3.fit(weather_train)
weather_train = scaler3.transform(weather_train)
weather_test = scaler3.transform(weather_test)
weather_pred = scaler3.transform(weather_pred)

#1.4 snack데이터
scaler4 = MinMaxScaler()
scaler4.fit(snack_train)
snack_train = scaler4.transform(snack_train)
snack_test = scaler4.transform(snack_test)
snack_pred = scaler4.transform(snack_pred)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input, LSTM

#2_1 모델1
input1 = Input(shape=(4,)) #input1 layer 구성
dense1_1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30, activation='relu')(dense1_1) 
dense1_3 = Dense(7, activation='relu')(dense1_2)
output1 = Dense(1)(dense1_3)

#2_2 모델2
input2 = Input(shape=(4,)) #input2 layer 구성
dense2_1 = Dense(100, activation='relu')(input2) 
dense2_2= Dense(30, activation='relu')(dense2_1) 
dense2_3= Dense(7, activation='relu')(dense2_2)
output2 = Dense(1)(dense2_3)

#2_3 모델3
input3 = Input(shape=(3,)) #input3 layer 구성
dense3_1 = Dense(100, activation='relu')(input3) 
dense3_2= Dense(30, activation='relu')(dense3_1) 
dense3_3= Dense(7, activation='relu')(dense3_2)
output3 = Dense(1)(dense3_3)

#2_4 모델4
input4 = Input(shape=(1,)) #input4 layer 구성
dense4_1 = Dense(100, activation='relu')(input4) 
dense4_2= Dense(30, activation='relu')(dense4_1) 
dense4_3= Dense(7, activation='relu')(dense4_2)
output4 = Dense(1)(dense4_3)

#모델 병합
merge1 = Concatenate()([output1, output2, output3, output4]) 

middle1 = Dense(30)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2)

#output 모델 구성 (분기-나눔)
output1_1 = Dense(30)(middle3)
output1_2 = Dense(13)(output1_1)
output1_3 = Dense(7)(output1_2)
output1_4 = Dense(1)(output1_3)

#모델 정의
model = Model(inputs=[input1, input2, input3, input4], outputs=output1_4)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
model.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='min') 
# check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min') #val_loss가 가장 좋은 값을 저장할 것이다
hist = model.fit([nc_train, am_train, weather_train, snack_train], y_train, epochs=1000, batch_size=32, validation_split=0.25, verbose=1, callbacks=[early_stopping])

# model.save_weights('./save/삼성시가/삼성시가_pred_model_weight.h5')

#4. 평가, 예측
loss = model.evaluate([nc_test, am_test, weather_test, snack_test], y_test, batch_size=32)
print("loss : ", loss)

y_pred = model.predict([nc_pred, am_pred, weather_pred, snack_pred])
print("2021년 최저시급 : ", y_pred)

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

# loss :  72062.640625
# 2021년 최저시급 :  [[4269.2646]]