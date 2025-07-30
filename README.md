# 🌎 Global Weather Forecasting with Machine Learning

**Predicting Global Temperature Using Environmental, Atmospheric, and Astronomical Features**

This project focuses on building robust machine learning models to predict temperature (in Fahrenheit) using a multi-domain dataset enriched with weather metrics, air quality indices, and astronomical data. The pipeline follows industry best practices, from data preprocessing to model tuning, validation, and performance comparison.

## 📘 Project Objective

To develop accurate regression models capable of forecasting temperature using real-time and historical weather data. Additionally, an exploratory classification task was attempted to predict weather conditions using logistic regression.

## 🧾 Dataset Overview

The dataset includes a wide array of features such as:

* **Geolocation**: latitude, longitude
* **Weather metrics**: temperature, wind speed and direction, pressure, humidity, cloud cover, visibility, gusts
* **Air Quality indices**: CO, NO₂, SO₂, PM2.5, PM10, Ozone
* **Astronomy**: sunrise/sunset, moonrise/moonset, moon illumination and phase
* **Timestamps**: decomposed into `day`, `month`, `year`, `hour`, and `minute`

## 🧹 Data Preprocessing & Feature Engineering

* **Dropped less relevant or redundant features** (e.g., `'temperature_celsius'`, `'timezone'`, `'country'`, `'location_name'`, `'feels_like_celsius'`, `'gust_kph'`, `'wind_kph'`)
* **Unified units** to avoid duplication (kept only **mph** and **miles** for wind and visibility)
* Converted timestamp fields into granular components: `last_day`, `last_month`, `last_year`, `last_hour`, `last_minute`
* Removed high-cardinality or uninformative categorical features (e.g., `'wind_direction'`, `'condition_text'`)
* Imputed missing values using **mean imputation**


## 🧠 Models Trained and Tuned

### ✅ Regressors (Trained with `RandomizedSearchCV` and Retested with Best Parameters):

| Model                   | Train R² | Train RMSE | Test R² | Test RMSE |
| ----------------------- | -------- | ---------- | ------- | --------- |
| Random Forest Regressor | 0.9951   | 1.4710     | 0.9617  | 11.4156   |
| Gradient Boosting       | 0.9960   | 1.1909     | 0.9709  | 8.6701    |
| XGBoost Regressor       | 0.9932   | 2.0339     | 0.9709  | 8.6797    |
| AdaBoost Regressor      | 0.7578   | 72.5146    | 0.7546  | 73.2018   |

### 🧪 Additional Experiments:

* **Linear Regression & SVM Regressor**:
  Implemented for benchmarking; showed lower R² and higher RMSE compared to ensemble models.

* **Logistic Regression on `condition_text`**:
  Attempted classification to predict weather conditions. Accuracy was not satisfactory, likely due to class imbalance and high categorical variance in conditions.


## 🔄 Cross-Validation (5-Fold)

To ensure generalizability and minimize overfitting, 5-fold cross-validation was applied on top-performing models:

### Gradient Boosting Regressor:

* Avg RMSE: `2.8597`
* Avg MAE: `2.0400`
* Avg R²: `0.9726`

### XGBoost Regressor:

* Avg RMSE: `2.8733`
* Avg MAE: `2.0703`
* Avg R²: `0.9724`


## 📊 Visualizations

* **Model Comparison Charts**: RMSE, MAE, and R² visualized for both training and test datasets across all models
* **Residual Plots**: Helped detect underfitting/overfitting tendencies and model biases


## 🛠 Technologies Used

* **Python 3.10+**
* **Scikit-learn**, **XGBoost**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**


## ✅ Key Insights

* **Gradient Boosting and XGBoost** emerged as the top-performing models with high R² and low prediction errors.
* **AdaBoost** did not generalize well and performed poorly compared to other ensemble models.
* **Logistic Regression** did not yield useful classification results for weather conditions due to class complexity.


## 🚀 Future Work

* Deploy top regression model as a **real-time weather prediction API**
* Build a **user-friendly front-end dashboard** for visualizing forecasts
* Retrain with **larger/more diverse datasets** and integrate **temporal modeling (e.g., LSTM)** for time-based prediction
