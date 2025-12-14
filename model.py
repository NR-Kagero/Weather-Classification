import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle
import xgboost as xgb

x_train = np.load("Data/extracted_features_train.npy")
x_test = np.load("Data/extracted_features_test.npy")
y_train = np.load("Data/labels_train.npy")
y_test = np.load("Data/labels_test.npy")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
xgClf=xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.5,reg_lambda=0.5,reg_alpha=0.1)
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial',penalty="l2",C=0.1, random_state=42)
clf.fit(x_train, y_train)
xgClf.fit(x_train,y_train)
y_pred_xg = xgClf.predict(x_test)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred_xg))
xgClf.save_model("xgb_model.json")
pickle.dump(clf, open('model.pkl', 'wb'))