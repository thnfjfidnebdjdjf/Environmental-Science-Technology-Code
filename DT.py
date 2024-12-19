import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

excel_file = r'data.xls'
data = pd.read_excel(excel_file, 'Sheet2')

data = data.dropna(subset=[data.columns[0]])

features = data.iloc[:, 1:11]
label = data.iloc[:, 11]

categorical_features = ['h5pmo10v2o40']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', features.columns.difference(categorical_features)),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

features.columns = features.columns.astype(str)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 6],
    'min_samples_leaf': [1, 2, 4, 5, 7],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['squared_error', 'friedman_mse']
}

dt_model = DecisionTreeRegressor()

grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')

grid_search.fit(features, label)

print("Best Parameters:", grid_search.best_params_)
print("Best Score (Negative MSE):", grid_search.best_score_)

best_model = grid_search.best_estimator_

best_model.fit(features, label)

y_pred = best_model.predict(features)

rmse = np.sqrt(mean_squared_error(label, y_pred))
r2 = r2_score(label, y_pred)

mape = np.mean(np.abs((label - y_pred) / label)) * 100

regression_slope, regression_intercept = np.polyfit(label, y_pred, 1)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

regression_slope_train, regression_intercept_train = np.polyfit(y_train, y_train_pred, 1)
regression_slope_test, regression_intercept_test = np.polyfit(y_test, y_test_pred, 1)

scatter_data_train = pd.DataFrame({'True Values': y_train, 'Predicted Values': y_train_pred})
scatter_data_test = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_test_pred})

scatter_excel_file_train = r"DT_train.xlsx"
scatter_excel_file_test = r"DT_test.xlsx"
scatter_data_train.to_excel(scatter_excel_file_train, index=False)
scatter_data_test.to_excel(scatter_excel_file_test, index=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7, label='True Values')
plt.plot(y_train, regression_slope_train * y_train + regression_intercept_train, color='red', label='Regression Line')
plt.title('Training Set Prediction Scatter Plot')
plt.xlabel('True Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')

plt.text(0.05, 0.9, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'R2 Score: {r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'MAPE: {mape:.2f}%', transform=plt.gca().transAxes)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7, label='True Values')
plt.plot(y_test, regression_slope_test * y_test + regression_intercept_test, color='red', label='Regression Line')
plt.title('Test Set Prediction Scatter Plot')
plt.xlabel('True Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')

plt.text(0.05, 0.9, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'R2 Score: {r2:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'MAPE: {mape:.2f}%', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
