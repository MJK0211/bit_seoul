#회귀, 리그레서 모델들을 추출

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1] #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor') #리그레서 모델들을 추출

for(name, algorithms) in allAlgorithms:
    try:
        kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
      
        model = algorithms()
        model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증된 scores를 모델과 엮어줌
        print(model, '의 정답률: ', scores)
    except:
        pass

# ARDRegression() 의 정답률:  [0.93770607 0.90490795 0.88150355 0.85813566 0.92066882]
# AdaBoostRegressor() 의 정답률:  [1.    1.    1.    1.    0.925]
# BaggingRegressor() 의 정답률:  [0.99375    0.98019802 1.         0.96652632 0.99936842]
# BayesianRidge() 의 정답률:  [0.95705201 0.93246259 0.9666229  0.92133256 0.94936182]
# CCA() 의 정답률:  [0.8798477  0.86181882 0.90823652 0.77901013 0.88643198]
# DecisionTreeRegressor() 의 정답률:  [1.         1.         1.         1.         0.93162393]
# DummyRegressor() 의 정답률:  [-0.00156685 -0.04526942 -0.34453125 -0.09869403 -0.18958914]
# ElasticNet() 의 정답률:  [0.84286327 0.88075163 0.86991493 0.9283455  0.88373968]
# ElasticNetCV() 의 정답률:  [0.90775704 0.92933895 0.95045328 0.94582915 0.97078795]
# ExtraTreeRegressor() 의 정답률:  [0.94392523 1.         1.         1.         0.86956522]
# ExtraTreesRegressor() 의 정답률:  [0.99361368 0.99356391 0.99922735 0.99948616 0.95529375]
# GaussianProcessRegressor() 의 정답률:  [ 0.28506181 -0.77747084  0.21562236  0.16663744  0.1568391 ]
# GradientBoostingRegressor() 의 정답률:  [1.         0.97484492 1.         1.         1.        ]
# HistGradientBoostingRegressor() 의 정답률:  [1.         1.         1.         0.92771084 0.93684532]
# HuberRegressor() 의 정답률:  [0.94048893 0.95743711 0.90023683 0.94650177 0.96850041]
# KNeighborsRegressor() 의 정답률:  [0.9875     0.98888889 0.99683168 0.99030303 0.95058824]
# KernelRidge() 의 정답률:  [0.94298633 0.94842827 0.93720345 0.96057581 0.93218702]
# Lars() 의 정답률:  [0.92251052 0.94576695 0.95383613 0.94622391 0.95675172]
# LarsCV() 의 정답률:  [0.93565222 0.94550934 0.9188773  0.95526393 0.96671211]
# Lasso() 의 정답률:  [0.91586729 0.85427839 0.84304126 0.89476425 0.89002723]
# LassoCV() 의 정답률:  [0.95218553 0.94479288 0.92294579 0.94906735 0.93812452]
# LassoLars() 의 정답률:  [-0.00074405 -0.01052632 -0.00752457 -0.00138206 -0.00065104]
# LassoLarsCV() 의 정답률:  [0.93971736 0.97544233 0.94646866 0.92202632 0.94161326]
# LassoLarsIC() 의 정답률:  [0.9470293  0.95144683 0.94589157 0.93831263 0.93396364]
# LinearRegression() 의 정답률:  [0.9600865  0.94557079 0.94724976 0.91767024 0.95735327]
# LinearSVR() 의 정답률:  [0.9226435  0.82114708 0.86971845 0.5989466  0.72477884]
# MLPRegressor() 의 정답률:  [-10.66201936 -19.62708167  -4.85579643 -11.85129299   0.95250613]
# NuSVR() 의 정답률:  [0.8841212  0.92479653 0.7371764  0.93122815 0.94899451]
# OrthogonalMatchingPursuit() 의 정답률:  [0.85599517 0.90986466 0.88786468 0.9398166  0.91228205]
# OrthogonalMatchingPursuitCV() 의 정답률:  [0.94498854 0.89489416 0.94940998 0.94832801 0.96585564]
# PLSCanonical() 의 정답률:  [0.35132915 0.20819927 0.3189009  0.42167325 0.55806626]
# PLSRegression() 의 정답률:  [0.90057679 0.93752539 0.88508312 0.95143006 0.9333465 ]
# PassiveAggressiveRegressor() 의 정답률:  [0.79036676 0.93046146 0.88728454 0.20510397 0.84408116]
# PoissonRegressor() 의 정답률:  [0.78254539 0.75903202 0.61910704 0.74678672 0.79435421]
# RANSACRegressor() 의 정답률:  [0.90722998 0.94645335 0.94451081 0.96036928 0.92610784]
# RadiusNeighborsRegressor() 의 정답률:  [-1.33333333 -0.81818182 -1.77894737 -2.40594059 -1.4735376 ]
# RandomForestRegressor() 의 정답률:  [0.99367391 0.99764337 0.9967913  0.95867524 0.99999443]
# Ridge() 의 정답률:  [0.94125109 0.93518639 0.96378244 0.93266976 0.95941372]
# RidgeCV(alphas=array([ 0.1,  1. , 10. ])) 의 정답률:  [0.95594477 0.93666482 0.96524553 0.912217   0.95209094]
# SGDRegressor() 의 정답률:  [-1.15247369e+23 -2.79062892e+24 -1.30633843e+26 -2.80868599e+26
#  -1.64894538e+26]
# SVR() 의 정답률:  [0.91488159 0.9349873  0.89759314 0.8437898  0.92138903]

