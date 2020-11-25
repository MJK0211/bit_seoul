
#다한 사람은 모델을 완성해서 결과 주석으로 적어놓을 것
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt

#1. 데이터
boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target , train_size=0.8, random_state=66, shuffle=True)

#2. 모델 구성

kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
parameters = [{"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
               "max_depth":[4, 5, 6]},
              {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
               "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]}, 
              {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.05],
               "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

#learning_rate가 가장 중요하다!

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=True) # SVC라는 모델을 GridSearchCV로 쓰겠다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc) 
print("최적의 매개변수 : ", model.best_estimator_) 

# acc :  0.93380326440394
# 최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=5,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=90, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)