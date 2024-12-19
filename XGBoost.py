import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

excel_file = r'data.xls'
data = pd.read_excel(excel_file, 'Sheet2')

data = data.dropna(subset=[data.columns[0]])

features = data.iloc[:, 1:11]
label = data.iloc[:, 11]

features.columns = features.columns.astype(str)

categorical_features = ['h5pmo10v2o40']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', features.columns.difference(categorical_features)),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

param_grid = {
    'learning_rate': [0.1, 0.2, 0.3, 0.4],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [1, 2, 3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

xgb_model = XGBRegressor()

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

xgb_model = XGBRegressor(**best_params)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)
print(f"10-Fold CV RMSE: {cv_rmse}")
print(f"10-Fold CV Average RMSE: {np.mean(cv_rmse):.2f}")

xgb_model.fit(X_train, y_train)

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

regression_slope_train, regression_intercept_train = np.polyfit(y_train, y_train_pred, 1)
regression_slope_test, regression_intercept_test = np.polyfit(y_test, y_test_pred, 1)

train_scatter_data = pd.DataFrame({'True Values': y_train, 'Train Predicted Values': y_train_pred})
test_scatter_data = pd.DataFrame({'True Values': y_test, 'Test Predicted Values': y_test_pred})

train_scatter_excel_file = r'XGBoost_train_predictions.xlsx'
test_scatter_excel_file = r'XGBoost_test_predictions.xlsx'

train_scatter_data.to_excel(train_scatter_excel_file, index=False)
test_scatter_data.to_excel(test_scatter_excel_file, index=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7, label='True Values')
plt.plot(y_train, regression_slope_train * y_train + regression_intercept_train, color='red', label='Regression Line')
plt.title('Train Prediction Scatter Plot')
plt.xlabel('True Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')
plt.text(0.05, 0.9, f'RMSE: {rmse_train:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'MAPE: {mape_train:.2f}%', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'R2 Score: {r2_train:.2f}', transform=plt.gca().transAxes)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7, label='True Values')
plt.plot(y_test, regression_slope_test * y_test + regression_intercept_test, color='red', label='Regression Line')
plt.title('Test Prediction Scatter Plot')
plt.xlabel('True Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')
plt.text(0.05, 0.9, f'RMSE: {rmse_test:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'MAPE: {mape_test:.2f}%', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'R2 Score: {r2_test:.2f}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
