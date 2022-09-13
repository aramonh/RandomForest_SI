import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


dataset = pd.read_csv('data/petrol_consumption.csv')

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
test_v = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=2000, random_state=0)
forest = regressor.fit(X_train, y_train)
#9.00,3571,1976,0.5250,541
y_pred = regressor.predict(X_test)

print("y_pred : \n", y_pred )

print('Mean Absolute Error : \n', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error : \n', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error : \n', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

