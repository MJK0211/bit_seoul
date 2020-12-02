import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

data_all = np.load('./project/data/npy/data_all.npy', allow_pickle=True)
count_list = np.load('./project/data/npy/count_list.npy', allow_pickle=True)
wage = np.load('./project/data/npy/wage.npy', allow_pickle=True)

#1. 데이터
data_year = np.asarray([])
for i in range(len(count_list)):
    if i == 0:
        data_year = data_all[i:count_list[i]]
    if i > 0 :
        data_year = np.vstack((data_year, data_all[count_list[i-1]:count_list[i]+count_list[i-1]]))

x_train, x_test, y_train, y_test = train_test_split(data_year, wage, train_size=0.9, random_state=66, shuffle=True)

#2. 모델 구성
n_estimators = 300
learning_rate = 1
colsample_bytree = 1
colsample_bylevel = 1

max_depth = 5
n_jobs = -1

model = XGBRegressor(max_depth = max_depth, learning_rate=learning_rate,
                     n_estimators = n_estimators, n_jobs=n_jobs,
                     colsample_bylevel = colsample_bylevel,
                     colsample_bytree = colsample_bytree)
#파라미터는 정답이 아니다
#파라미터 공부하기!
#속도차이가 많이 나기 때문에 xgbooster가 더 좋다고 평가할 수 있다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print("acc : ", acc)
print(model.feature_importances_)
