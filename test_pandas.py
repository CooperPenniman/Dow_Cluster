# Import pandas and cars.csv
import pandas as pd

from sklearn.svm import LinearSVC






data = pd.read_csv("train.csv")

X_train = data['Fare'].values
y_train = data ['Survived'].values

passid=data['PassengerId'].values

print(passid)

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_train)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)



