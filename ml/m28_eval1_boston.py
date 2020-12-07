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
# model = XGBRegressor(n_estimators=1000, learning_rate=0.1)
model = XGBRegressor(learning_rate=0.1)

#3. 컴파일, 훈련
model.fit(x_train, y_train, verbose=True, eval_metric="rmse", eval_set=[(x_test, y_test), (x_test, y_test)]) 
#평가를 명시해줘야 훈련 모습을 보여준다.
#eval_metric="rmse"
#eval_set=[(x_test, y_test)]

#eval_metric = "rmse", "mae", "logloss", "error", "auc"

results = model.evals_result()
print(len(results))
print("eval's results : ", results['validation_0']['rmse'][-1:])

#4. 평가, 예측
score = model.score(x_test, y_test)
print("R2 : ", score)


# 결과값
#              ...
# [988]   validation_0-rmse:2.55719
# [989]   validation_0-rmse:2.55719
# [990]   validation_0-rmse:2.55719
# [991]   validation_0-rmse:2.55719
# [992]   validation_0-rmse:2.55719
# [993]   validation_0-rmse:2.55719
# [994]   validation_0-rmse:2.55719
# [995]   validation_0-rmse:2.55719
# [996]   validation_0-rmse:2.55719
# [997]   validation_0-rmse:2.55719
# [998]   validation_0-rmse:2.55719
# [999]   validation_0-rmse:2.55719
# R2 :  0.9108295653765306