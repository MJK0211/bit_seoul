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
model.fit(x_train, y_train, verbose=True, eval_metric="mae", eval_set=[(x_test, y_test), (x_test, y_test)]) 
#평가를 명시해줘야 훈련 모습을 보여준다.
#eval_metric="rmse"
#eval_set=[(x_test, y_test)]

#eval_metric = "rmse", "mae", "logloss", "error", "auc"

results = model.evals_result()
print(len(results))
print("eval's results : ", results['validation_0']['mae'][-1:])

# 2
# eval's results :  [1.884314]

#4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("R2 : ", r2)


# 결과값
#                                    ...
# [91]    validation_0-mae:1.88659        validation_1-mae:1.88659
# [92]    validation_0-mae:1.88671        validation_1-mae:1.88671
# [93]    validation_0-mae:1.88919        validation_1-mae:1.88919
# [94]    validation_0-mae:1.88576        validation_1-mae:1.88576
# [95]    validation_0-mae:1.88955        validation_1-mae:1.88955
# [96]    validation_0-mae:1.88713        validation_1-mae:1.88713
# [97]    validation_0-mae:1.88443        validation_1-mae:1.88443
# [98]    validation_0-mae:1.88368        validation_1-mae:1.88368
# [99]    validation_0-mae:1.88431        validation_1-mae:1.88431
# R2 :  0.8884947020459273