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

#2. 모델 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = RandomForestClassifier(n_estimators=100)

#3. 훈련
model.fit(x_train_minmax, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test_minmax)
print(y_test[20:30].reshape(10,), "의 예측결과 : \n", y_pred[20:30])
score = model.score(x_test_minmax, y_test)
print("score : ", score)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", acc_score)

# RandomForestClassfier()
# [1 0 1 1 1 1 1 1 1 1] 의 예측결과 :
#  [1 1 1 1 1 1 1 1 1 1]
# score :  0.9479591836734694
# acc_score :  0.9479591836734694