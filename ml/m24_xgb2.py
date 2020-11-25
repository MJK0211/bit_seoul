
#다한 사람은 모델을 완성해서 결과 주석으로 적어놓을 것
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터
boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target , train_size=0.8, random_state=66, shuffle=True)

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

plot_importance(model) #xgbooster에서는 plot_importance를 제공해준다
plt.show()

# 결과값
# acc :  0.8454028763099724
# [2.39500515e-02 1.11327665e-02 2.00374089e-02 1.09804654e-03
#  4.44661006e-02 1.87476233e-01 1.26504647e-02 5.00670299e-02
#  7.12032706e-05 1.53776342e-02 3.83814536e-02 1.33807389e-02
#  5.81910908e-01]