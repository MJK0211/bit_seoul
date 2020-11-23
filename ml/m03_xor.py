import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. 모델
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측

y_pred = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_pred)

score = model.score(x_data, y_data)
print("score : ", score)

acc_score = accuracy_score(y_data, y_pred)
print("acc_score : ", acc_score)

# 결과값
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  [1 1 1 1]
# score :  0.5
# acc_score :  0.5