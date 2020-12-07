from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel #SelectFromModel 추가!

data_all = np.load('./project/data/npy/data_all.npy', allow_pickle=True)
count_list = np.load('./project/data/npy/count_list.npy', allow_pickle=True)
wage = np.load('./project/data/npy/wage.npy', allow_pickle=True)

#1. 데이터
data_year = list()


for i in range(len(count_list)):
    if i == 0:
        subset = np.array(data_all[i : count_list[i]])       
        print(np.average(subset, axis=0))
        data_year = np.array(subset)      
    if i == 1 :     
        subset = np.array(data_all[count_list[i-1] : (count_list[i-1] + count_list[i])])
        data_year = np.append(data_year, [subset])

# print(data_all)
# for i in range(len(count_list)):
#     if i == 0:        
#         subset = data_all[i : count_list[i]]
#         data_year.append([subset])      
#     if i > 0 :     
#         subset = data_all[count_list[i-1] : (count_list[i-1] + count_list[i])]
#         data_year.append(subset)
        
# x_train, x_test, y_train, y_test = train_test_split(data_year, wage, train_size=0.9, random_state=66, shuffle=True)

data_year = np.array(data_year)

# print(data_year.shape) #(10,)
# print(data_year[0][0].shape) #(10,)
# print(data_year[1].shape) #(10,)

# print(wage)

#2. 모델 구성
# kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
# parameters = [{"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
#                "max_depth":[4, 5, 6]},
#               {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
#                "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]}, 
#               {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.05],
#                "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}


# model = GridSearchCV(XGBClassifier(n_jobs=-1), parameters, cv=kfold, verbose=True)
model = XGBClassifier(n_jobs=-1)

#3. 훈련
model.fit(data_year, wage)

#4. 평가, 예측
acc = model.score(data_year, wage)

# print("acc : ", acc)
# print(model.feature_importances_)
