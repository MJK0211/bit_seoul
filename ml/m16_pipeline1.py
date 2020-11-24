#pipeline

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline, make_pipeline #Pipeline, make_pipeline 추가 - Scaler를 엮어보자
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler #Scaler 4개 추가

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 


#1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1] #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)

#pipe = make_pipeline(MaxAbsScaler(), SVC()) #Scaler를 사용할 때, crossvalidation 과적합을 피하기위해 묶어줌 (소문자 pipeline)

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())]) #(대문자 pipeline) 'svm'부분을 name으로 수정가능
pipe.fit(x_train, y_train)
print('acc : ', pipe.score(x_test, y_test))

# MinMaxScaler
# acc :  1.0

# StandardScaler
# acc :  0.9666666666666667

# RobustScaler
# acc :  0.9666666666666667

# MaxAbsScaler
# acc :  1.0

'''
#2. 모델 구성

kfold = KFold(n_splits=5, shuffle=True) 

parameters = [{"C": [1, 10, 100, 1000], "kernel": ["linear"], 
               "C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma":[0.001, 0.0001],
               "C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma":[0.001, 0.0001]}         
]

model = RandomizedSearchCV(SVC(), parameters, cv=kfold) #n_iter - 기본값 = 10

#3. 훈련
model.fit(x, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_) #파라미터중에 어떤 모델이 좋은지 가장 좋은 결과값을 리턴해준다, best_estimator = 가장좋은 평가자

y_pred = model.predict(x_test_minmax)

print("y_pred : ", y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("최종 정답률 : ", accuracy_score)

'''