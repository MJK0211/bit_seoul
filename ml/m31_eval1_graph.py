#eval = evaluate
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel #SelectFromModel 추가!
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.8, random_state=42, shuffle=True)

#2. 모델 구성
model = XGBRegressor(n_estimators = 1000, learning_rate=0.1)

#3. 컴파일, 훈련
model.fit(x_train, y_train, verbose=True, eval_metric=['logloss', 'rmse'], eval_set=[(x_train, y_train), (x_test, y_test)]) 

results = model.evals_result()
print(len(results))
print("eval's results : ", results['validation_0']['rmse'][-1:])

# 2
# eval's results :  [0.33108]

#4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("R2 : ", r2)

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
# ax.legend()
# plt.ylabel("LogLoss")
# plt.title("XGBoost LogLoss")
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel("Rmse")
plt.title("XGBoost RMSE")
plt.show()

