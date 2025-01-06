import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import time
import sys

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
    return data
    # return np.log2(data)
    # return np.right_shift(data, 20)
    # return np.divide(data, (1 << 20))

def restore_predict(data):
    return data
    # return np.exp2(data)
    # return np.left_shift(data, 20)
    # return np.multiply(data, (1 << 20))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
# Output the min, max, and mean of 'memCostBytes' in dataset
def print_dataset_stats(y_test):
    stats = y_test.describe()
    print(stats)


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
    dnew = xgb.DMatrix(test_data.filter(regex=filter_regex))
    y_pred = model.predict(dnew)
    y_pred = restore_predict(y_pred)
    y_new_true = test_data[predict_value]
    evaluate_predict_result(y_pred, y_new_true)
    
def evaluate_model(dtest, y_true, model):
    y_pred = model.predict(dtest)
    evaluate_predict_result(y_pred, y_true)

# Training params
predict_value = 'cpuCostNs'
train_dataset_file = sys.argv[1]
filter_regex = f'^(feature_|table_|{predict_value})'
features_regex = f'^(feature_|table_)'
train_data_hold_ratio = 0.8
evaluate_data_ratio = 0.2
num_rounds = 100

# Model params
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',  # or 'rmse'
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10,
    'gamma': 2,
    'alpha': 1
}
    
# Load the dataset
data = pd.read_csv(train_dataset_file)
# Apply one hot encoding on tables
data = pd.get_dummies(data, columns=data.filter(regex='^table_').columns)


## Data cleanup
# Filter out rows where predict_value is 0
data = data[data[predict_value] != 0]

# Assume the samples follow the standard distribution, drop the exceptional data
# grouped = data.groupby('digest')[predict_value]
# std = grouped.transform('std')
# avg = grouped.transform('mean')
# data = data[abs(data[predict_value] - avg) <= 3 * std]

# Split the dataset into two parts according to digest
digest_values = data['digest'].unique()
np.random.shuffle(digest_values)
train_digest_values = digest_values[:int(train_data_hold_ratio * len(digest_values))]
train_data = data[data['digest'].isin(train_digest_values)]
test_data = data[~data['digest'].isin(train_digest_values)]

# Data preprocessing for train data
X = train_data.filter(regex=features_regex)
y = train_data[predict_value]

# Data preprocessing for test data
X_test_data, y_test_data = test_data.filter(regex=features_regex), test_data[predict_value]
X_test_data = transform_predict(X_test_data)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=evaluate_data_ratio, random_state=42)
y_train = transform_predict(y_train)
y_test = transform_predict(y_test)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the model
evals = [(dtrain, 'train'), (dtest, 'test')]
start_time = time.time()
model = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds=10)
end_time = time.time()
training_time = end_time - start_time
print(f"Model training time: {training_time:.3f} seconds")

# Save the model
model.save_model('xgboost_sql_model.json')
print("Finish training model, saved to file")
print("==========================================")

# Evaluate the model
evaluate_model(dtest, y_test, model)
print_dataset_stats(restore_predict(y_test))

# Print the most significant features and their relevancy
feature_importance = model.get_fscore()
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("==========================================")
print("Most significant features and their relevancy:")
for feature, importance in feature_importance[:10]:  # Select top 10
    print(f"{feature}: {importance}")
    # stats = X[feature].describe()
    # print(f"Statistics for feature '{feature}':\n{stats}")
print("==========================================")

# Example: Load the model and make predictions
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_sql_model.json')
dnew = xgb.DMatrix(X_test)
y_new_pred = loaded_model.predict(dnew)

# Predict using the loaded model
print("==========================================")
print("evaluate the model on a brand-new test set")
print_dataset_stats(test_data[predict_value])

X_dtest_data = xgb.DMatrix(X_test_data, label=y_test_data)
evaluate_model(X_dtest_data, y_test_data, model)
