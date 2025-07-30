#Reading the dataset using pandas
import pandas as pd
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

data = data.drop(['country','location_name','timezone','wind_direction','moon_phase','condition_text','feels_like_fahrenheit'],axis =1)
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

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size= 0.30,random_state = 42)

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

#USing support vector regression
from sklearn.svm import SVR

svr = SVR()

#Fitting the model
svr.fit(X_train,y_train)

y_pred = svr.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(root_mean_squared_error(y_test, y_pred))

#Checking train accuracy 
y_predict = svr.predict(X_train)

from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

print(r2_score(y_train, y_predict))
print(mean_absolute_error(y_train, y_predict))
print(root_mean_squared_error(y_train, y_predict))



'''If we were to use the hyperparameter tuning then it would take a lot of time
so not tuning the model further for minimal increase in accuracy'''

#Hyperparameter tuning using GridSearchCV

#Defining parameter range 

params = {'C':[0.1],
          'gamma':[1,0],
          'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVR(), param_grid = params,refit= True, verbose = 3)

grid.fit(X_train,y_train)

grid.best_params_

grid_predict = grid.predict(X_test)
































