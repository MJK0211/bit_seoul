# 클래스파이어 모델들 추출
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

x_train = np.load('./data/npy/boston_x_train.npy')
y_train = np.load('./data/npy/boston_y_train.npy')

x_test = np.load('./data/npy/boston_x_test.npy')
y_test = np.load('./data/npy/boston_y_test.npy')

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor') #리그레서 모델들을 추출

for(name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', r2_score(y_test, y_pred))
    except:
        pass
import sklearn
print(sklearn.__version__)# 0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

# ARDRegression 의 정답률:  0.7127233456747222
# AdaBoostRegressor 의 정답률:  0.8286531474742342
# BaggingRegressor 의 정답률:  0.8548548222179326
# BayesianRidge 의 정답률:  0.7630459331021768
# CCA 의 정답률:  0.6531018118621725
# DecisionTreeRegressor 의 정답률:  0.5944595546376428
# DummyRegressor 의 정답률:  -0.022137487754498864
# ElasticNet 의 정답률:  0.7167441474214005
# ElasticNetCV 의 정답률:  0.7040653241054833
# ExtraTreeRegressor 의 정답률:  0.7321100103898588
# ExtraTreesRegressor 의 정답률:  0.8885195915011297
# GammaRegressor 의 정답률:  -0.022137487754498864
# GaussianProcessRegressor 의 정답률:  -8.815748760459142
# GradientBoostingRegressor 의 정답률:  0.883366022228985
# HistGradientBoostingRegressor 의 정답률:  0.9038362067962236
# HuberRegressor 의 정답률:  0.8086584505257499
# KNeighborsRegressor 의 정답률:  0.5874732301133119
# KernelRidge 의 정답률:  0.7861691338567625
# Lars 의 정답률:  0.7237704162260382
# LarsCV 의 정답률:  0.7491156328619468
# Lasso 의 정답률:  0.7127696231707341
# LassoCV 의 정답률:  0.7340154235278986
# LassoLars 의 정답률:  -0.022137487754498864
# LassoLarsCV 의 정답률:  0.7436922850027454
# LassoLarsIC 의 정답률:  0.7498613618847099
# LinearRegression 의 정답률:  0.7436922850027438
# LinearSVR 의 정답률:  0.7559084137605632
# MLPRegressor 의 정답률:  0.5902850029678458
# NuSVR 의 정답률:  0.3383872428898339
# OrthogonalMatchingPursuit 의 정답률:  0.581205917466397
# OrthogonalMatchingPursuitCV 의 정답률:  0.6989268971115848
# PLSCanonical 의 정답률:  -4.050786072082147
# PLSRegression 의 정답률:  0.7241658915958643
# PassiveAggressiveRegressor 의 정답률:  -2.302808371121875
# PoissonRegressor 의 정답률:  0.8309464312159538
# RANSACRegressor 의 정답률:  0.3084949114867229
# RandomForestRegressor 의 정답률:  0.854024530775289
# Ridge 의 정답률:  0.7603766785081536
# RidgeCV 의 정답률:  0.7473686660672407
# SGDRegressor 의 정답률:  -2.4461092611752384e+26
# SVR 의 정답률:  0.33564193604100967
# TheilSenRegressor 의 정답률:  0.7943447479724708
# TransformedTargetRegressor 의 정답률:  0.7436922850027438
# TweedieRegressor 의 정답률:  0.7295168974543231
# 0.23.1