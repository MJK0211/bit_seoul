#eval = evaluate
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel #SelectFromModel 추가!
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.8, random_state=42, shuffle=True)

print(x_train.shape) #(120, 4)
print(y_train.shape) #(120,)
print(x_test.shape) #(30, 4)
print(y_test.shape) #(30,)

#2. 모델 구성

model = XGBClassifier(n_estimators = 1000, learning_rate=0.01)
#3. 컴파일, 훈련
model.fit(x_train, y_train, verbose=True, eval_metric="auc", eval_set=[(x_test, y_test)]) 

results = model.evals_result()
print(len(results))
print("eval's results : ", results['validation_0']['auc'][-1:])

# 1
# eval's results :  [0.991811]

# #4. 평가, 예측
y_predict = model.predict(x_test)

acc_score = accuracy_score(y_test, y_predict)
print("acc_score : ", acc_score)

# acc_score :  0.9649122807017544