import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

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

# log transformation: memory / 16MB
def transform_predict(data):
    return np.log2(data)
    # return np.right_shift(data, 20)
    # return np.divide(data, (1 << 20))

def restore_predict(data):
    return np.exp2(data)
    # return np.left_shift(data, 20)
    # return np.multiply(data, (1 << 20))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
# Output the min, max, and mean of 'memCostBytes' in dataset
def print_dataset_stats(y_test):
    min_mem_cost = y_test.min()
    max_mem_cost = y_test.max()
    mean_mem_cost = y_test.mean()
    stddev_mem_cost = y_test.std()
    print(f"""
          Min: {bytes_to_human_readable(min_mem_cost)}, 
          Max: {bytes_to_human_readable(max_mem_cost)}, 
          Mean: {bytes_to_human_readable(mean_mem_cost)}, 
          Stddev: {bytes_to_human_readable(stddev_mem_cost)}
          """)


def evaluate_predict_result(y_pred, y_test):
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mae_ratio = mae / y_pred.mean()
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # print(f"RMSE on test set: {bytes_to_human_readable(rmse)}")
    # print(f"MAE on test set: {bytes_to_human_readable(mae)}")
    # print(f"MAE on test set: {(mae)}")
    # print(f"MAE/MEAN on test set: {mae_ratio:.3f}")
    print(f"MAPE on test set: {mape:.3f}%")
   
    
def evaluate_model_from_raw_data(test_data, model):
    dnew = xgb.DMatrix(test_data.drop(columns=['cpuCostNs', 'memCostBytes', 'digest']))
    y_pred = model.predict(dnew)
    y_pred = restore_predict(y_pred)
    y_new_true = test_data['memCostBytes']
    evaluate_predict_result(y_pred, y_new_true)
    
def evaluate_model(dtest, y_true, model):
    y_pred = model.predict(dtest)
    # y_pred = restore_predict(y_pred)
    evaluate_predict_result(y_pred, y_true)

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
    'eval_metric': 'mape',  # or 'rmse'
    'eta': 0.1,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10,
    'gamma': 2,
    'alpha': 1
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = transform_predict(y_train)
y_test = transform_predict(y_test)

# Convert to DMatrix format, the efficient data format recommended by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the model
num_rounds = 200
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds=10)

# Evaluate the model
evaluate_model(dtest, y_test, model)

print_dataset_stats(restore_predict(y_test))

# Save the model
model.save_model('xgboost_sql_model.json')

# Example: Load the model and make predictions
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_sql_model.json')
dnew = xgb.DMatrix(X_test)
y_new_pred = loaded_model.predict(dnew)

# Predict using the loaded model
test_data = pd.read_csv('test_data.csv')
print("==========================================")
print("evaluate the model on a brand-new test set")
print_dataset_stats(test_data['memCostBytes'])
evaluate_model_from_raw_data(test_data, model)