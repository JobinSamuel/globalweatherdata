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

data['condition_text'].value_counts()

# Defining a frequency threshold
threshold = 20

# Getting value counts
condition_counts = data['condition_text'].value_counts()

# Identifying rare conditions
rare_conditions = condition_counts[condition_counts < threshold].index

# Replacing them with "Rare"
data['condition_text_grouped'] = data['condition_text'].apply(lambda x: 'Rare' if x in rare_conditions else x)

#Creating a label encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['condition_label'] = le.fit_transform(data['condition_text_grouped'])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)


data = data.drop(['country','location_name','timezone','wind_direction','moon_phase','condition_text_grouped','condition_text','feels_like_fahrenheit'],axis =1)
data.info()

#Checking correlation 

corr = data.corr(numeric_only=True)
corr


# Plot the heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(corr, linewidths=0.5, fmt=".2f")
plt.title('Correlation Heatmap (excluding NaNs)')
plt.show()


data['temperature_fahrenheit'].describe()

sns.histplot(data['temperature_fahrenheit'], bins=30, kde=True)


#Splitting the data into train and test 

X = data.drop(columns = ['condition_label'])

y = data['condition_label']

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

#Using logistic regression

from sklearn.linear_model import LogisticRegression

#Since there are 40+ classes changed the default maximum iteration from 100 to 1000
logistic = LogisticRegression(max_iter = 1000)

logistic.fit(X_train,y_train)

y_pred = logistic.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accu_score = accuracy_score(y_test, y_pred)
print(accu_score)

#Accuracy remains the same even after using randomizedcv it must because of the multiple climatic conditions
penalty = ['l1','l2','elasticnet']

c_values = [100,10,0.1,1.0,0.01]

solver = ['newton-cg','lbfgs','liblinear','sag','saga']

params = dict(penalty = penalty,C = c_values, solver = solver)

from sklearn.model_selection import RandomizedSearchCV

model = LogisticRegression()

randomcv = RandomizedSearchCV(estimator = model, param_distributions= params, cv = 5, scoring = 'accuracy')


randomcv.fit(X_train,y_train)

randomcv.best_params_
randomcv.best_score_

y_predi = randomcv.predict(X_test)
y_predi

from sklearn.metrics import accuracy_score

accu_score = accuracy_score(y_test, y_predi)
print(accu_score)






'''Tried hyperparameter tuning for the model Logistic Regression with 62,000 records 
and 40+ classes + GridSearchCV which is slow and prone to convergence warnings.
'''

''' Hyperparameter Tuning using Gridsearch CV
penalty = ['l1','l2','elasticnet']

c_values = [100,10,0.1,1.0,0.01]

solver = ['newton-cg','lbfgs','liblinear','sag','saga']

params = dict(penalty = penalty,C = c_values, solver = solver)

#Hyperparameter tuning using GridsearchCV to find the best parameters

from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold() 

grid = GridSearchCV(estimator = logistic, param_grid = params, scoring = 'accuracy', cv = cv, n_jobs= -1)

grid.best_params_'''

'''Hyperparamter tuning using RandomizedsearchCV
from sklearn.model_selection import RandomizedSearchCV
model = LogisticRegression()
randomcv = RandomizedSearchCV(estimator = model, param_distributions= params, cv = 5, scoring = accuracy)

randomcv.best_params_
randomcv.best_score_'''














