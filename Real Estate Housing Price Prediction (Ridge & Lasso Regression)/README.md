## Real Estate Housing Price Prediction using Ridge and Lasso Regression


### Detailed Problem Statement & Requirements
- Predict median house values using the California Housing dataset. 
- Engineer interaction features, impute missing values, handle multicollinearity, and compare Ordinary Least Squares (OLS), Ridge (L2), and Lasso (L1) regression models based on RMSE, MAE, and R² scores.

### Suggested Technical Approach
1. Load California Housing dataset. 
2. Impute null values and compute correlation heatmaps. 
3. Standardize continuous variables and encode categorical attributes. 
4. Fit Ridge and Lasso models across multiple alpha hyperparameters using GridSearchCV. 
5. Plot predicted vs actual values and residual distribution curves.

### Tools:
Python
Scikit-Learn (LinearRegression, Ridge, Lasso)
GridSearchCV