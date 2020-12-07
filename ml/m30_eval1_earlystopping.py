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
model = XGBRegressor(n_estimators = 3000, learning_rate=0.1)

#3. 컴파일, 훈련
model.fit(x_train, y_train, verbose=True, eval_metric="rmse", eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=20) 

results = model.evals_result()
print(len(results))
print("eval's results : ", results['validation_0']['rmse'][-1:])

# 2
# eval's results :  [0.33108]

#4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_pred, y_test)
print("R2 : ", r2)

# 결과값
#                                    ...
# [117]   validation_0-rmse:0.36705       validation_1-rmse:2.56753
# [118]   validation_0-rmse:0.36332       validation_1-rmse:2.56734
# [119]   validation_0-rmse:0.35322       validation_1-rmse:2.56665
# [120]   validation_0-rmse:0.34801       validation_1-rmse:2.56768
# [121]   validation_0-rmse:0.34374       validation_1-rmse:2.56683
# [122]   validation_0-rmse:0.34048       validation_1-rmse:2.56697
# [123]   validation_0-rmse:0.33724       validation_1-rmse:2.56814
# [124]   validation_0-rmse:0.33108       validation_1-rmse:2.56709
# [125]   validation_0-rmse:0.32701       validation_1-rmse:2.56809
# Stopping. Best iteration:
# [105]   validation_0-rmse:0.41718       validation_1-rmse:2.56245  -> validation_1이 기준

# R2 :  0.8887083149507902

