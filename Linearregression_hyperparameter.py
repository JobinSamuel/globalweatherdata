#Reading the dataset using pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/jobinsamuel/Desktop/datascienceprojects/GlobalWeatherRepository.csv")
#Checking if the data was loaded properly
data.head()

#Checking the total number of rows and columns in the dataset
data.shape

#Checking if there are any NA or null values in the dataset
data.isnull().sum()

#Checking if there are any duplicated values in the dataset
data[data.duplicated()]

data.info()

data['last_updated'] = pd.to_datetime(data['last_updated'],format = '%Y-%m-%d %H:%M')

data['last_day'] = data['last_updated'].dt.day
data['last_month'] = data['last_updated'].dt.month
data['last_year'] = data['last_updated'].dt.year

data['last_hour']= data['last_updated'].dt.hour
data['last_min']= data['last_updated'].dt.minute

data.info()

data = data.drop(['last_updated_epoch','last_updated','temperature_celsius','wind_kph','pressure_mb','precip_mm',
                  'feels_like_celsius','visibility_km','gust_kph'],axis = 1)

data['sunrise'] = pd.to_datetime(data['sunrise'], format = '%I:%M %p').dt.strftime('%H:%M')
data['sunset'] = pd.to_datetime(data['sunset'], format = '%I:%M %p').dt.strftime('%H:%M')


data['sunrise_hour'] = data['sunrise'].str.split(':').str[0].astype('int')
data['sunrise_min'] = data['sunrise'].str.split(':').str[1].astype('int')

data['sunset_hour'] = data['sunset'].str.split(':').str[0].astype('int')
data['sunset_min'] = data['sunset'].str.split(':').str[1].astype('int')


data['moonrise'] = data['moonrise'].replace("No moonrise", pd.NA)
data['moonrise'] = pd.to_datetime(data['moonrise'], format='%I:%M %p', errors='coerce')


data['moonset'] = data['moonset'].replace("No moonset", pd.NA)
data['moonset'] = pd.to_datetime(data['moonset'], format='%I:%M %p', errors='coerce')

data['moonrise_hour'] = data['moonrise'].apply(lambda x: x.hour if pd.notna(x) else pd.NA).astype('Int64')
data['moonrise_minute'] = data['moonrise'].apply(lambda x: x.minute if pd.notna(x) else pd.NA).astype('Int64')

data['moonset_hour'] = data['moonset'].apply(lambda x: x.hour if pd.notna(x) else pd.NA).astype('Int64')
data['moonset_minute'] = data['moonset'].apply(lambda x: x.minute if pd.notna(x) else pd.NA).astype('Int64')

data = data.drop(['sunrise','sunset','moonrise','moonset'],axis = 1)

data.info()

data = data.drop(['country','location_name','timezone','condition_text','wind_direction','moon_phase'],axis =1)
data.info()

sns.histplot(data['visibility_miles'], bins=30, kde=True)
plt.title('Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()


sns.histplot(data['air_quality_Carbon_Monoxide'], bins=30, kde=True)
plt.title('Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

#Checking correlation 
corr_ = data.corr()

corr = data.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(corr, linewidths=0.5)
plt.title('Correlation Heatmap (excluding NaNs)')
plt.show()


data['temperature_fahrenheit'].describe()
sns.histplot(data['temperature_fahrenheit'], bins=30, kde=True)

data
#Splitting the data into train and test 

X = data.drop(columns = ['temperature_fahrenheit'])

y = data['temperature_fahrenheit']


from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size= 0.30,random_state = 101)

#Imputing NA values with mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

# Fiting only on training data
X_train = imputer.fit_transform(X_train)

# Applying the same transformation to test data
X_test = imputer.transform(X_test)

#Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

#Reading the dataset using pandas
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




#Lasso Regression (L1 Regularization) - Used for Feature selection 
# it has lambda and slopw but when the lambda becomes 0 then that feature 
#is considered as less significant and will be removed.

#Using Lasso Crossvalidation to get the optimal alpha value

from sklearn.linear_model import LassoCV
lassocv = LassoCV(cv =5)
lassocv.fit(X_train,y_train)
lassocv_pred = lassocv.predict(X_test)

lassocv_mae = mean_absolute_error(y_test, lassocv_pred)
lassocv_r2 = r2_score(y_test, lassocv_pred)
print(lassocv_mae)
print(lassocv_r2)
lassocv.alpha_ #Shows which alpha was chosen
lassocv.alphas_ #Shows all the 100alphas which were used


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1044) 
'''without passing an alpha value i got an accuracy of 97.1%
but after using alpha = 0.1044 the accuracy improved to 97.7%'''
lasso.fit(X_train,y_train)
lasso_y_pred = lasso.predict(X_test)

lasso_mae = mean_absolute_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)
print(lasso_mae)
print(lasso_r2)

'''Ridge Regression (L2 Regularization) - Normally used to reduce Overfitting if any
it has lambda and slope but the lambda will never be 0 

Using Ridge Crossvalidation to get the optimal alpha value'''

from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(cv =5)
ridgecv.fit(X_train,y_train)
ridgecv_pred = ridgecv.predict(X_test)

ridgecv_mae = mean_absolute_error(y_test, ridgecv_pred)
ridgecv_r2 = r2_score(y_test,ridgecv_pred)
print(ridgecv_mae)
print(ridgecv_r2)

ridgecv.alpha_ #The chosen alpha value 
ridgecv.alphas #These were the alphas that were used 

from sklearn.linear_model import Ridge
'''Without using alpha I got an accuracy of 97.65% 
the accuracy increased slightly by 0.1%'''

ridge = Ridge(alpha= 0.1)
ridge.fit(X_train,y_train)
ridge_y_pred = ridge.predict(X_test)


ridge_mae = mean_absolute_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)
print(ridge_mae)
print(ridge_r2)

'''ElasticNetRegression is a combination of both l1 and l2 regularization where it  
reduces overfitiing and can be used for feature selection. We use these
techniques to hyperparameter tuning the Linear Regression '''

#Using ElasticNet Cross Validation to get the optimal alpha value
from sklearn.linear_model import ElasticNetCV

elasticv = ElasticNetCV(cv = 5)
elasticv.fit(X_train,y_train)
elasticv_pred = elasticv.predict(X_test)

elasticv_mae = mean_absolute_error(y_test, elasticv_pred)
elasticv_r2 = r2_score(y_test,elasticv_pred)
print(elasticv_mae)
print(elasticv_r2)

elasticv.alpha_
elasticv.alphas_

from sklearn.linear_model import ElasticNet
'''The accuracy without using alpha value is 87.82% after adding the alpha = 0.0340
The accuracy increased to 97.59%'''

elastic = ElasticNet(alpha= 0.0340)
elastic.fit(X_train,y_train)
elastic_y_pred = elastic.predict(X_test)


elastic_mae = mean_absolute_error(y_test, elastic_y_pred)
elastic_r2 = r2_score(y_test, elastic_y_pred)
print(elastic_mae)
print(elastic_r2)

