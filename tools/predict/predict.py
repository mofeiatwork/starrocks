import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Load the dataset
# Assuming the dataset is a CSV file containing feature columns and target column 'execution_time'
data = pd.read_csv('sql_features.csv')

# Data preprocessing
# Assuming the target column is 'execution_time' and the other columns are features
X = data.drop(columns=['cpuCostNs', 'memCostBytes'])
y = data['memCostBytes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformation on both training and testing sets
# X_train = np.right_shift(X_train, 10) 
# X_test = np.right_shift(X_test, 10) 
# y_train = np.right_shift(y_train, 10)
# y_test = np.right_shift(y_test, 10)
# X_train = np.log(X_train) 
# X_test = np.log(X_test) 
# y_train = np.log(y_train)
# y_test = np.log(y_test)


# Convert to DMatrix format, the efficient data format recommended by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
# params = {
#     'objective': 'reg:squarederror',  # Regression task
#     'eval_metric': 'mae',           # Use mean absolute error as evaluation metric
#     'eta': 0.1,                      # Learning rate
#     'max_depth': 6,                  # Maximum depth of the tree
#     'subsample': 0.8,                # Subsample ratio
#     'colsample_bytree': 0.8          # Subsample ratio of columns
# }

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

# Train the model
num_rounds = 100
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds=10)

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
print(f"RMSE on test set: {bytes_to_human_readable(rmse)}")
print(f"MAE on test set: {bytes_to_human_readable(mae)}")

# Save the model
model.save_model('xgboost_sql_model.json')

# Example: Load the model and make predictions
# loaded_model = xgb.Booster()
# loaded_model.load_model('xgboost_sql_model.json')
# dnew = xgb.DMatrix(X_test)
# y_new_pred = loaded_model.predict(dnew)
