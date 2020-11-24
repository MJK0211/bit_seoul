#grid_Search

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #GridSearchCV 추가
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1] #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다

parameters = [{"C": [1, 10, 100, 1000], "kernel": ["linear"], # 1번: 1-linear 2번: 10-lenear, 3번: 100-leanear 4번: 100-leanear - 4번
               "C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma":[0.001, 0.0001], # rbf 당 감마 4 * 2 = 8번
               "C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma":[0.001, 0.0001]} # sigmoid 당 감마 4 * 2 = 8번

            #   SVC는 kernel parameter를 가지고 있는데 어떤 kernel을 사용하는지에 따라 관련있는 parameter들이 결정된다.
            #   kernel='linear'이면 C parameter만 사용하고
            #   kernel='rbf'이면 C와 gamma를 모두 사용한다.
] #총 20번 * 5 (kfold) = 총 100번!

# model = SVC()
model = GridSearchCV(SVC(), parameters, cv=kfold) # SVC라는 모델을 GridSearchCV로 쓰겠다
#하이퍼파라미터 20번 * 크로스 발리데이션 5번 = 100번

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_) #파라미터중에 어떤 모델이 좋은지 가장 좋은 결과값을 리턴해준다, best_estimator = 가장좋은 평가자

y_pred = model.predict(x_test)

print("y_pred : ", y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("최종 정답률 : ", accuracy_score)

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# 최종 정답률 :  1.0