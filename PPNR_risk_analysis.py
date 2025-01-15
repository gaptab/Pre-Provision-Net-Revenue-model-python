# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Generate Dummy Data for PPNR Models
np.random.seed(42)
dates = pd.date_range(start="2018-01-01", periods=60, freq='ME')
revenue = np.random.uniform(100, 500, len(dates))
economic_indicators = np.random.uniform(50, 150, len(dates))
management_overlay = np.random.uniform(0.8, 1.2, len(dates))

data = pd.DataFrame({
    'Date': dates,
    'Revenue': revenue,
    'Economic_Indicator': economic_indicators,
    'Management_Overlay': management_overlay,
    'Adjusted_Revenue': revenue * management_overlay
})

# 2. Apply OLS Regression
X = data[['Economic_Indicator']].values
y = data['Adjusted_Revenue'].values

ols_model = LinearRegression()
ols_model.fit(X, y)
y_pred_ols = ols_model.predict(X)

# Metrics for OLS
ols_r2 = r2_score(y, y_pred_ols)
ols_rmse = np.sqrt(mean_squared_error(y, y_pred_ols))

# 3. Apply ARIMAX Model
exog = data['Economic_Indicator']
endog = data['Adjusted_Revenue']

arimax_model = ARIMA(endog, order=(1, 1, 1), exog=exog)
arimax_fit = arimax_model.fit()
arimax_forecast = arimax_fit.predict(start=50, end=59, exog=exog[50:])

# 4. Backtesting
actual = data['Adjusted_Revenue'][50:60]
backtesting_mse = mean_squared_error(actual, arimax_forecast)

# 5. Sensitivity Analysis
sensitivity_scenarios = {
    'Scenario': ['Optimistic', 'Pessimistic', 'Neutral'],
    'Economic_Indicator_Factor': [1.1, 0.9, 1.0]
}

sensitivity_df = pd.DataFrame(sensitivity_scenarios)
sensitivity_df['Revenue_Forecast'] = sensitivity_df['Economic_Indicator_Factor'] * ols_model.coef_[0] * economic_indicators.mean() + ols_model.intercept_

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Adjusted_Revenue'], label='Actual Revenue')
plt.plot(data['Date'], y_pred_ols, label='OLS Prediction', linestyle='--')
plt.title('OLS Regression vs Actual Revenue')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['Date'][50:60], actual, label='Actual Revenue')
plt.plot(data['Date'][50:60], arimax_forecast, label='ARIMAX Forecast', linestyle='--')
plt.title('ARIMAX Backtesting')
plt.legend()
plt.show()

# Print Outputs
print("OLS Regression Metrics:")
print(f"R-squared: {ols_r2:.2f}, RMSE: {ols_rmse:.2f}")

print("\nARIMAX Backtesting:")
print(f"Mean Squared Error: {backtesting_mse:.2f}")

print("\nSensitivity Analysis Results:")
print(sensitivity_df)

