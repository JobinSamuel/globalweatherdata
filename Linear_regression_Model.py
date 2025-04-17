#Reading the dataset using pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Building a Linear Regression model
from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(X_train,y_train)

#Checking the intercept and coefficient values 
print(regression.intercept_)

print(regression.coef_)

from sklearn.model_selection import cross_val_score

crossvalscore = cross_val_score(regression, X_train,y_train,scoring = 'neg_mean_squared_error', cv = 3)
np.mean(crossvalscore)

#Prediction 
y_pred = regression.predict(X_test) 

y_train_pred = regression.predict(X_train)

#printing the RMSE, MSE, MAE and R2 score on test data 
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

test_mse = mean_squared_error(y_test, y_pred)
print(test_mse)

test_mae = mean_absolute_error(y_test, y_pred)
print(test_mae)

test_rmse = np.sqrt(test_mse)
print(test_rmse)

test_score = r2_score(y_test, y_pred)
print(test_score)

#printing the RMSE, MSE, MAE and R2 score on training data 

train_mse = mean_squared_error(y_train, y_train_pred)
print(train_mse)

train_mae = mean_absolute_error(y_train, y_train_pred)
print(train_mae)

train_rmse = np.sqrt(train_mse)
print(train_rmse)

train_r2score = r2_score(y_train, y_train_pred)
print(train_r2score)

#Calculating the adjusted R2 for test data
print(1-(1-test_score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

plt.scatter(y_test,y_pred)

residuals = y_test - y_pred
print(residuals)

sns.displot(residuals, kind ='kde')


#Polynomial transformation

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

from sklearn.linear_model import LinearRegression

poly_regression = LinearRegression()

poly_regression.fit(X_train, y_train)

poly_y_pred = poly_regression.predict(X_test)

poly_y_train_pred = poly_regression.predict(X_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

poly_test_mse = mean_squared_error(y_test, poly_y_pred)
print(poly_test_mse)

poly_test_mae = mean_absolute_error(y_test, poly_y_pred)
print(poly_test_mae)

poly_test_rmse = np.sqrt(poly_test_mse)
print(poly_test_rmse)

poly_test_score = r2_score(y_test, poly_y_pred)
print(poly_test_score)

#printing the RMSE, MSE, MAE and R2 score on training data 

poly_train_mse = mean_squared_error(y_train, poly_y_train_pred)
print(poly_train_mse)

poly_train_mae = mean_absolute_error(y_train, poly_y_train_pred)
print(poly_train_mae)

poly_train_rmse = np.sqrt(poly_train_mse)
print(poly_train_rmse)

poly_train_r2score = r2_score(y_train, poly_y_train_pred)
print(poly_train_r2score)


residuals = y_test - poly_y_pred
print(residuals)

sns.displot(residuals, kind ='kde')

plt.figure(figsize=(8,6))
plt.scatter(y_test, poly_y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.xlabel("Actual Temperature (°F)")
plt.ylabel("Predicted Temperature (°F)")
plt.title("Actual vs Predicted Temperatures")
plt.grid(True)
plt.tight_layout()
plt.show()
