from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier                          

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size = 0.8, shuffle = True, random_state = 66)
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)

model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

# model.save_model('./model/xgb/cancer.xgb.model')

model2 = XGBClassifier() #모델2는 클래스파이어 명시를 해주고
model2.load_model('./model/xgb/cancer.xgb.model')

y_predict = model2.predict(x_test)
acc2 = accuracy_score(y_test, y_predict)

print("acc2 : ", acc2)
print(model.feature_importances_)

# acc2 :  0.9736842105263158
# acc2 :  0.9736842105263158