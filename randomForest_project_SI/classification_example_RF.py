import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from itertools import permutations , product



dataset = pd.read_csv('data/bill_authentication.csv')

print(dataset.head())

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train", X_train)
print("X_test", X_test)
print("y_train",y_train)
print("y_test",y_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classificator = RandomForestClassifier(n_estimators=200, random_state=0)
forest = classificator.fit(X_train, y_train)

y_pred = classificator.predict(X_test)

print("y_pred : \n", y_pred )


print("CONFUSION MATTIX : \n", confusion_matrix(y_test,y_pred))
print("CLASSIFICATION REPORT : \n",classification_report(y_test,y_pred))
print("ACCURACY SCORE : \n", accuracy_score(y_test, y_pred))

