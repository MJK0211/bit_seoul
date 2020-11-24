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

# pipe = make_pipeline(MinMaxScaler(), SVC()) #Scaler를 사용할 때, crossvalidation 과적합을 피하기위해 묶어줌
pipe = Pipeline([("scaler", MinMaxScaler()), ('malddong', SVC())])
#2. 모델 구성

kfold = KFold(n_splits=5, shuffle=True) 

# parameters = [{"svc__C": [1, 10, 100, 1000], "svc__kernel": ["linear"], 
#                "svc__C": [1, 10, 100, 1000], "svc__kernel": ["rbf"], "svc__gamma":[0.001, 0.0001],
#                "svc__C": [1, 10, 100, 1000], "svc__kernel": ["sigmoid"], "svc__gamma":[0.001, 0.0001]}         
# ]
parameters = [{"malddong__C": [1, 10, 100, 1000], "malddong__kernel": ["linear"], 
               "malddong__C": [1, 10, 100, 1000], "malddong__kernel": ["rbf"], "malddong__gamma":[0.001, 0.0001],
               "malddong__C": [1, 10, 100, 1000], "malddong__kernel": ["sigmoid"], "malddong__gamma":[0.001, 0.0001]}         
]
model = RandomizedSearchCV(pipe, parameters, cv=kfold) #n_iter - 기본값 = 10

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('acc : ', model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_) #파라미터중에 어떤 모델이 좋은지 가장 좋은 결과값을 리턴해준다, best_estimator = 가장좋은 평가자

# acc :  1.0
# 최적의 매개변수 :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('svc', SVC(C=1000, gamma=0.001, kernel='sigmoid'))])

# 대문자 Pipeline - malddong 수정
# acc :  1.0
# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('malddong', SVC(C=1000, gamma=0.001, kernel='sigmoid'))])