Global Weather Forecasting with Machine Learning
This project explores the application of Machine Learning techniques to predict temperature (in Fahrenheit) using rich, multi-featured global weather data. The data includes variables such as wind, visibility, air quality metrics, sun and moon phases, and more.

So far, this project features:

Data preprocessing and transformation

Feature engineering for date/time and astronomical fields

One-hot encoding for categorical features

Linear Regression and Polynomial Regression models for baseline prediction

Error metrics evaluation (MAE, MSE, RMSE, R², Adjusted R²)

Visualizations for residuals and correlation heatmaps

 Coming Soon: Additional ML models including Decision Trees, Random Forests, Gradient Boosting, Support Vector Machines, and Neural Networks to compare performance and improve accuracy!

 Dataset
The dataset used is GlobalWeatherRepository.csv, which contains worldwide weather readings along with timestamps, air quality indicators, and astronomical information like sunrise, sunset, moonrise, and moonset.

Key Features Extracted:
Temporal: year, month, day, hour, minute

Astronomical: sunrise/sunset & moonrise/moonset times (converted to hour/minute format)

Air Quality: carbon monoxide, ozone, nitrogen dioxide, etc.

Categorical Encoded: country, location_name, timezone, wind_direction, condition_text, moon_phase

 Models Used
Linear Regression
Simple baseline to model relationships between features and temperature

Metrics: MAE, MSE, RMSE, R², Adjusted R²

Polynomial Regression
Explores non-linearity in the data with interaction terms

Uses PolynomialFeatures() from sklearn to transform input space

Evaluation Metrics
Each model is evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score and Adjusted R²

Residuals are also visualized using KDE plots to inspect model bias.

Visualizations
Histograms with KDE for feature distributions

Correlation heatmap

Actual vs Predicted scatter plots

These plots help with:

Understanding feature importance and data distribution

Detecting overfitting and variance issues

Upcoming Work
This project will be extended to include and compare various ML algorithms:

 Decision Tree Regression

 Random Forest Regression

 XGBoost/Gradient Boosting

 Support Vector Regression

 KNN Regression

 Deep Learning Models (Keras/PyTorch)

 Goal
To identify the most accurate and generalizable model for weather forecasting based on global weather metrics an important step toward building smarter, ML-driven climate systems and urban planning tools.


