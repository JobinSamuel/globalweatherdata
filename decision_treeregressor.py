#Reading the dataset using pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/jobinsamuel/Desktop/datascienceprojects/GlobalWeatherRepository.csv")
#Checking if the data was loaded properly
data.head()
data.columns
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
data.value_counts('wind_direction')
data = data.drop(['country','location_name','timezone','wind_direction','moon_phase','condition_text','feels_like_fahrenheit'],axis =1)
data.info()

data.columns

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

corr = data.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(corr, linewidths=0.5, fmt=".2f")
plt.title('Correlation Heatmap (excluding NaNs)')
plt.show()

data['temperature_fahrenheit'].describe()
sns.histplot(data['temperature_fahrenheit'], bins=30, kde=True)


#Splitting the data into train and test 

X = data.drop(columns = ['temperature_fahrenheit'])

y = data['temperature_fahrenheit']


from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size= 0.25,random_state = 42)

#Imputing NA values with mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

# Fiting only on training data
X_train = imputer.fit_transform(X_train)

# Applying the same transformation to test data
X_test = imputer.transform(X_test)

#Using decision tree regression     
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

#Fitting the model 
dtr.fit(X_train,y_train)

#Using the model to predict 
y_pred = dtr.predict(X_test)

#Checking the r2 score, mae, rmse
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(root_mean_squared_error(y_test, y_pred))


''''When checking the training accuracy the model is overfitting giving a 100% 
so need to preprune the decision tree'''
#Checking train accuracy 
y_predict = dtr.predict(X_train)

from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

print(r2_score(y_train, y_predict))
print(mean_absolute_error(y_train, y_predict))
print(root_mean_squared_error(y_train, y_predict))


#Hyperparameter tuning model 

'''Observed that feautures like sqrt and log2 in hyperparameter tuning were taking 
too much time'''
params = { 
    'criterion': ['squared_error','friedman_mse','absolute_error'],
    'splitter': ['best','random'],
    'max_depth': [8,9,10,11,12,13,14],
    'max_features': ['auto','sqrt','log2']
    }
#Using GridsearchCV
from sklearn.model_selection import GridSearchCV

ddtr = DecisionTreeRegressor()

grid = GridSearchCV(ddtr, param_grid = params,cv = 6, verbose = 3)

grid.fit(X_train,y_train)

y_preprune_pred = grid.predict(X_test)


print(r2_score(y_test,y_preprune_pred))
print(mean_absolute_error(y_test,y_preprune_pred))
print(root_mean_squared_error(y_test, y_preprune_pred))

y_pretrain_pred = grid.predict(X_train)


print(r2_score(y_train, y_pretrain_pred))
print(mean_absolute_error(y_train, y_pretrain_pred))
print(root_mean_squared_error(y_train, y_pretrain_pred))
grid.best_params_

grid.best_score_

'''Now after pruning the model is not overfitting the train accuracy is 95% and 
the test accuracy is 86.4% and so we can say that the model is not overfitting'''

#Plotting the top features for predicting temperature
import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.Series(grid.best_estimator_.feature_importances_, 
                               index=X.columns).sort_values(ascending=False)
feature_importance.head(15).plot(kind='barh', figsize=(8,6))
plt.title("Top Features Predicting Temperature")
plt.show()


