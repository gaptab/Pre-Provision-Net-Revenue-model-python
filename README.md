# PPNR-model

![alt text](https://github.com/gaptab/Pre-Provision-Net-Revenue-model-python/blob/main/OLS_regression_vs_actual_revenue.png)
![alt text](https://github.com/gaptab/Pre-Provision-Net-Revenue-model-python/blob/main/ARIMAX_backtesting.png)



Step 1: Generate Dummy Data
Created a dataset of Revenue, Economic_Indicator, and Management_Overlay with random values to simulate real-world PPNR data.
Calculated Adjusted_Revenue as Revenue * Management_Overlay.

Step 2: Apply OLS Regression
Used Economic_Indicator as the independent variable to predict Adjusted_Revenue.
Evaluated the model's performance using R-squared and RMSE.

Step 3: Apply ARIMAX Model
Built an ARIMAX model with Economic_Indicator as an exogenous variable and Adjusted_Revenue as the endogenous variable.
Predicted future values for backtesting.

Step 4: Backtesting
Compared ARIMAX model predictions with actual revenue for a specific period.
Calculated Mean Squared Error (MSE) as the evaluation metric.

Step 5: Sensitivity Analysis
Simulated scenarios (Optimistic, Pessimistic, Neutral) by varying the Economic_Indicator.
Predicted revenue forecasts for each scenario using the OLS model.

Visualizations
OLS Regression vs. Actual Revenue: Compared predicted and actual revenue values.
ARIMAX Backtesting: Visualized forecasted vs actual values for the testing period.

Outputs

OLS Regression Metrics: R-squared and RMSE values to assess the linear regression model.
ARIMAX Backtesting: MSE to evaluate the ARIMAX model's forecasting accuracy.
Sensitivity Analysis: Forecasted revenue under different economic scenarios.
