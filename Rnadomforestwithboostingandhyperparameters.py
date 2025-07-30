#Reading the dataset using pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Creating a function to evaluate the model 

def evaluate_models(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_sc = r2_score(true, predicted)
    return mae,mse,r2_sc
    
models = {
    
    "Random Forest Regressor": RandomForestRegressor(),
    "Adaboost Regressor":AdaBoostRegressor(),
    "Graident BoostRegressor":GradientBoostingRegressor(),
    "Xgboost Regressor":XGBRegressor()
   
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)    
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
# Evaluate Train and Test dataset
    model_train_mae,model_train_rmse,model_train_r2 = evaluate_models(y_train, y_train_pred)

    model_test_mae,model_test_rmse,model_test_r2 = evaluate_models(y_test, y_test_pred)
    
    print(list(models.keys())[i])
    
    
    print("Model Performance for training set")
    print("-Root Mean Squared Error :{:.4f}".format(model_train_rmse))
    print("-Mean Absolute Error :{:.4f}".format(model_train_mae))
    print("-R2 Score:{:.4f}".format(model_train_r2))
    
    print('----------------------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))

    print('='*35)
    print('\n')
    
#Initialize few parameter for Hyperparamter tuning

rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

ada_params={"n_estimators":[50,60,70,80],
    "loss":['linear','square','exponential']}

gradient_params={"loss": ['squared_error','huber','absolute_error'],
             "criterion": ['friedman_mse','squared_error','mse'],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500],
              "max_depth": [5, 8, 15, None, 10]}
              
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}

    
# Models list for Hyperparameter tuning
randomcv_models = [("RF", RandomForestRegressor(), rf_params),
                   ("Adaboost",AdaBoostRegressor(),ada_params),
                   ("Gradient Boosting",GradientBoostingRegressor(),gradient_params),
                   ("XGboost",XGBRegressor(),xgboost_params)
                   
                   ]    
    
##Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=3,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])    
    
## Retraining the models with best parameters
models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators = 1000, min_samples_split= 2, max_features= 8, max_depth = None),
     "Adaboost Regressor": AdaBoostRegressor(n_estimators = 60, loss= 'square'),
     "Gradient Boosting": GradientBoostingRegressor(n_estimators = 500, min_samples_split = 20, max_depth = 8, loss = 'squared_error', criterion = 'squared_error'),
     "Xgboost Regressor":XGBRegressor(n_estimators =  300, max_depth = 8, learning_rate = 0.1, colsample_bytree = 0.8)
    
}    
    
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_models(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_models(y_test, y_test_pred)
    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    
    print('='*35)
    print('\n')
'''Model Comparison Summary (after hyperparameter tuning using RandomizedSearchCV):
- Random Forest Regressor showed excellent performance with high R² on both t
raining (0.9951) and test (0.9617) sets, but signs of slight overfitting due to
 a larger gap in RMSE.
- Gradient Boosting and XGBoost both demonstrated strong generalization with 
nearly identical results (Test R² ≈ 0.9709), making them top contenders 
for deployment based on predictive accuracy and robustness.
- AdaBoost significantly underperformed with lower R² and high RMSE, indicating 
underfitting and limited usefulness for this task.

Overall, Gradient Boosting and XGBoost are preferred for their balance of accuracy,
 generalization, and training stability.'''
    
from sklearn.model_selection import cross_val_score, KFold

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Select the model
model = models["Gradient Boosting"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = cross_val_score(model, X_imputed, y, scoring='neg_root_mean_squared_error', cv=kfold)
mae_scores = cross_val_score(model, X_imputed, y, scoring='neg_mean_absolute_error', cv=kfold)
r2_scores = cross_val_score(model, X_imputed, y, scoring='r2', cv=kfold)

print("Cross-Validation Results (5-Fold):")
print(f"- Avg RMSE: {np.mean(-rmse_scores):.4f}")
print(f"- Avg MAE: {np.mean(-mae_scores):.4f}")
print(f"- Avg R2 Score: {np.mean(r2_scores):.4f}")


model = models["Xgboost Regressor"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = cross_val_score(model, X_imputed, y, scoring='neg_root_mean_squared_error', cv=kfold)
mae_scores = cross_val_score(model, X_imputed, y, scoring='neg_mean_absolute_error', cv=kfold)
r2_scores = cross_val_score(model, X_imputed, y, scoring='r2', cv=kfold)

print("Cross-Validation Results (5-Fold):")
print(f"- Avg RMSE: {np.mean(-rmse_scores):.4f}")
print(f"- Avg MAE: {np.mean(-mae_scores):.4f}")
print(f"- Avg R2 Score: {np.mean(r2_scores):.4f}")