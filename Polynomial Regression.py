import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv('assignment1_dataset.csv')
print(data.describe())
data.dropna(how='any', inplace=True)
X = data.iloc[:, 0:2]  # age , bmi, children
Y = data['medical_cost']
cols = ('Age', 'Bmi', 'Children')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=2)
poly_features = PolynomialFeatures(degree=3)
X_train_poly=poly_features.fit_transform(X_train)
poly_model=linear_model.LinearRegression(normalize=True)
#poly_model.normalize=True
poly_model.fit(X_train_poly,y_train)
y_train_predicted=poly_model.predict(X_train_poly)
mdeicalCost_Prediction=poly_model.predict(poly_features.fit_transform(X_test))
print("MSE = ",metrics.mean_squared_error(y_test,mdeicalCost_Prediction))
#print("Accuracy = ",metrics.accuracy_score(y_test,mdeicalCost_Prediction,normalize=False))
