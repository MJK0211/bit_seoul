# 클래스파이어 모델들 추출

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier') #클래스파이어 모델들을 추출

for(name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', accuracy_score(y_test, y_pred))
    except:
        pass
import sklearn
print(sklearn.__version__)# 0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함

# AdaBoostClassifier 의 정답률:  0.6333333333333333
# BaggingClassifier 의 정답률:  0.9666666666666667
# BernoulliNB 의 정답률:  0.3
# CalibratedClassifierCV 의 정답률:  0.9
# CategoricalNB 의 정답률:  0.9
# ComplementNB 의 정답률:  0.6666666666666666
# DecisionTreeClassifier 의 정답률:  0.9333333333333333
# DummyClassifier 의 정답률:  0.26666666666666666
# ExtraTreeClassifier 의 정답률:  0.9
# ExtraTreesClassifier 의 정답률:  0.9666666666666667
# GaussianNB 의 정답률:  0.9666666666666667
# GaussianProcessClassifier 의 정답률:  0.9666666666666667
# GradientBoostingClassifier 의 정답률:  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률:  0.8666666666666667
# KNeighborsClassifier 의 정답률:  0.9666666666666667
# LabelPropagation 의 정답률:  0.9333333333333333
# LabelSpreading 의 정답률:  0.9333333333333333
# LinearDiscriminantAnalysis 의 정답률:  1.0
# LinearSVC 의 정답률:  0.9666666666666667
# LogisticRegression 의 정답률:  1.0
# LogisticRegressionCV 의 정답률:  1.0
# MLPClassifier 의 정답률:  1.0
# MultinomialNB 의 정답률:  0.9666666666666667
# NearestCentroid 의 정답률:  0.9333333333333333
# NuSVC 의 정답률:  0.9666666666666667
# PassiveAggressiveClassifier 의 정답률:  0.7
# Perceptron 의 정답률:  0.9333333333333333
# QuadraticDiscriminantAnalysis 의 정답률:  1.0
# RadiusNeighborsClassifier 의 정답률:  0.9666666666666667
# RandomForestClassifier 의 정답률:  0.9666666666666667
# RidgeClassifier 의 정답률:  0.8666666666666667
# RidgeClassifierCV 의 정답률:  0.8666666666666667
# SGDClassifier 의 정답률:  0.9
# SVC 의 정답률:  0.9666666666666667
# 0.23.1

