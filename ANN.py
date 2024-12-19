import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

excel_file = r'data.xls'
data = pd.read_excel(excel_file, 'Sheet2')

data = data.dropna(subset=[data.columns[0]])

features = data.iloc[:, 1:11]
label = data.iloc[:, 11]

categorical_feature = ['h5pmo10v2o40']

encoder = OneHotEncoder(drop='first', sparse=False)
categorical_data = encoder.fit_transform(data[categorical_feature])
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names(categorical_feature))

data = data.drop(categorical_feature, axis=1)

features.columns = features.columns.astype(str)


def build_and_train_model(hidden_units, activation_function, learning_rate, epochs):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=features.shape[1], activation=activation_function))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, model


hidden_units_range = list(range(20, 30))
activation_functions = ['relu']
learning_rate = 0.0003
epochs = 1000

best_rmse = float('inf')
best_r2 = -float('inf')
best_hidden_units = None
best_activation_function = None

kf = KFold(n_splits=10, shuffle=True, random_state=42)

mean_rmse_list = []
mean_r2_list = []

for hidden_units in hidden_units_range:
    for activation_function in activation_functions:
        fold_rmse_list = []
        fold_r2_list = []

        for train_index, test_index in kf.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = label.iloc[train_index], label.iloc[test_index]

            rmse, r2, model = build_and_train_model(hidden_units, activation_function, learning_rate, epochs)

            fold_rmse_list.append(rmse)
            fold_r2_list.append(r2)

        mean_rmse = np.mean(fold_rmse_list)
        mean_r2 = np.mean(fold_r2_list)

        print(f"Hidden Units: {hidden_units}, Activation Function: {activation_function}",
              f"Mean RMSE: {mean_rmse:.6f}, Mean R2 Score: {mean_r2:.6f}")

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_r2 = mean_r2
            best_hidden_units = hidden_units
            best_activation_function = activation_function

print("Best Parameters:")
print(f"Hidden Units: {best_hidden_units}")
print(f"Activation Function: {best_activation_function}")
print(f"Best Mean RMSE: {best_rmse:.2f}")
print(f"Best Mean R2 Score: {best_r2:.6f}")

final_rmse, final_r2, final_model = build_and_train_model(best_hidden_units, best_activation_function, learning_rate,
                                                          epochs)
print("Final Model Performance:")
print(f"RMSE: {final_rmse:.2f}")
print(f"R2 Score: {final_r2:.6f}")

y_pred_train = final_model.predict(X_train).flatten()
y_pred_test = final_model.predict(X_test).flatten()

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

regression_slope_train, regression_intercept_train = np.polyfit(y_train, y_pred_train, 1)
regression_slope_test, regression_intercept_test = np.polyfit(y_test, y_pred_test, 1)

scatter_data_train = pd.DataFrame({'True Values': y_train, 'Predicted Values': y_pred_train})
scatter_data_test = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_test})

scatter_excel_file_train = r'ANN_Optimized_Parameters_train.xlsx'
scatter_excel_file_test = r'ANN_Optimized_Parameters_test.xlsx'
scatter_data_train.to_excel(scatter_excel_file_train, index=False)
scatter_data_test.to_excel(scatter_excel_file_test, index=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.7, label='True Values')
plt.plot(y_train, regression_slope_train * y_train + regression_intercept_train, color='red', label='Regression Line')
plt.title('Training Set Prediction Scatter Plot')
plt.xlabel('True Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')

plt.text(0.05, 0.9, f'RMSE: {rmse_train:.2f}', transform=plt.gca().transAxes, )
plt.text(0.05, 0.85, f'MAPE: {mape_train:.2f}%', transform=plt.gca().transAxes, )
plt.text(0.05, 0.8, f'R2 Score: {r2_train:.6f}', transform=plt.gca().transAxes, )

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.7, label='True Values')
plt.plot(y_test, regression_slope_test * y_test + regression_intercept_test, color='red', label='Regression Line')
plt.title('Test Set Prediction Scatter Plot')
plt.xlabel('True Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')

plt.text(0.05, 0.9, f'RMSE: {rmse_test:.2f}', transform=plt.gca().transAxes, )
plt.text(0.05, 0.85, f'MAPE: {mape_test:.2f}%', transform=plt.gca().transAxes, )
plt.text(0.05, 0.8, f'R2 Score: {r2_test:.6f}', transform=plt.gca().transAxes, )

plt.tight_layout()
plt.show()
