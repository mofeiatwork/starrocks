import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Load the dataset
# Assuming the dataset is a CSV file containing feature columns and target column 'execution_time'
data = pd.read_csv('sql_features.csv')

# Data preprocessing
# Assuming the target column is 'execution_time' and the other columns are features
X = data.drop(columns=['cpuCostNs', 'memCostBytes', 'digest'])
y = data['memCostBytes']

# Define the parameters for the model
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',  # or 'rmse'
    'eta': 0.1,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10,
    'alpha': 1
}

# Initialize the KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the list to store the results
results = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert to DMatrix format, the efficient data format recommended by XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train the model
    num_rounds = 20
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds=10)

    # Evaluate the model
    y_pred = model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    results.append(mae)

# Predict
y_pred = model.predict(dtest)


# Output the min, max, and mean of 'memCostBytes' in X_train
min_mem_cost = y_train.min()
max_mem_cost = y_train.max()
mean_mem_cost = y_train.mean()
stddev_mem_cost = y_train.std()

# Convert bytes to human-readable format
def bytes_to_human_readable(bytes):
    if bytes < 1024:
        return f"{bytes} bytes"
    elif bytes < 1048576:
        return f"{bytes/1024:.2f} KB"
    elif bytes < 1073741824:
        return f"{bytes/1048576:.2f} MB"
    elif bytes < 1099511627776:
        return f"{bytes/1073741824:.2f} GB"
    else:
        return f"{bytes/1099511627776:.2f} TB"

print(f"Min of 'memCostBytes' in Y_train: {bytes_to_human_readable(min_mem_cost)}")
print(f"Max of 'memCostBytes' in Y_train: {bytes_to_human_readable(max_mem_cost)}")
print(f"Mean of 'memCostBytes' in Y_train: {bytes_to_human_readable(mean_mem_cost)}")
print(f"Stddev of 'memCostBytes' in Y_train: {bytes_to_human_readable(stddev_mem_cost)}")

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE on train set: {bytes_to_human_readable(rmse)}")
print(f"MAE on train set: {bytes_to_human_readable(mae)}")

# Save the model
model.save_model('xgboost_sql_model.json')

# Example: Load the model and make predictions
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_sql_model.json')
dnew = xgb.DMatrix(X_test)
y_new_pred = loaded_model.predict(dnew)

# Load a new test dataset from file 'test_data.csv'
test_data = pd.read_csv('test_data.csv')
# Predict using the loaded model
dnew = xgb.DMatrix(test_data.drop(columns=['cpuCostNs', 'memCostBytes', 'digest']))
y_new_pred = loaded_model.predict(dnew)
# Evaluate the RMSE and MAE
y_new_true = test_data['memCostBytes']
rmse = np.sqrt(mean_squared_error(y_new_true, y_new_pred))
mae = mean_absolute_error(y_new_true, y_new_pred)
print(f"RMSE on test set: {bytes_to_human_readable(rmse)}")
print(f"MAE on test set: {bytes_to_human_readable(mae)}")
