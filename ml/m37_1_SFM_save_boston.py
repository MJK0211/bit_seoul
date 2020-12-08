from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier                          

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size = 0.8, shuffle = True, random_state = 66)

models = [DecisionTreeClassifier(max_depth=4), RandomForestClassifier(max_depth=4), GradientBoostingClassifier(max_depth=4), XGBClassifier(max_depth=4)]

for i in range(len(models)):

    model = models[i]
    print(model)
    model.fit(x_train, y_train)        
    # y_predict = model.predict(x_test)
    # acc = accuracy_score(y_test, y_predict)
    # print("acc2 : ", acc)
    # # print(model.feature_importances_)

    # model.save_model('./model/xgb/'+ model + '.model')
    

# model2 = XGBClassifier() #모델2는 클래스파이어 명시를 해주고
# model2.load_model('./model/xgb/cancer.xgb.model')

# y_predict = model2.predict(x_test)
# acc2 = accuracy_score(y_test, y_predict)

# print("acc2 : ", acc2)
# print(model.feature_importances_)

# # acc2 :  0.9736842105263158
# # acc2 :  0.9736842105631588
