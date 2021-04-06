import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from math import floor


## Project-Part1
def construct_training_feature_matrix(past_cases_interval, past_weather_interval):
    columns = []
    weather_features = ['max_temp', 'max_dew', 'max_humid']
    for weather_feature in weather_features:
        for j in range(past_weather_interval):
            feature = f'{weather_feature}-{past_weather_interval - j}'
            columns.append(feature)
    for i in range(past_cases_interval):
        feature = f'past_cases-{past_cases_interval - i}'
        columns.append(feature)
    training_feature_matrix = pd.DataFrame(columns=columns)  # 一个空的数据框，之后会往其中加入数据以构建训练集
    return training_feature_matrix

def insert_value_into_training_feature_matrix(training_feature_matrix, past_weather_interval, past_cases_interval):
    num_of_columns_of_matrix = past_weather_interval * 3 + past_cases_interval
    row = []
    asc_of_weather = -1
    asc_of_case = -1
    for i in range(31, 193):
        for j in range(num_of_columns_of_matrix):
            if j < past_weather_interval:
                column_name = 'max_temp'
            elif j >= past_weather_interval and j < 2 * past_weather_interval:
                column_name = 'max_dew'
            elif j >= 2 * past_weather_interval and j < 3 * past_weather_interval:
                column_name = 'max_humid'
            else:
                column_name = 'dailly_cases'
            if j < num_of_columns_of_matrix - past_cases_interval:
                row.append(train_df.loc[i - past_weather_interval + asc_of_weather, column_name])
            else:
                row.append(train_df.loc[i - past_weather_interval + asc_of_case, column_name])

            asc_of_weather += 1
            if asc_of_weather == past_weather_interval - 1:
                asc_of_weather = -1
            asc_of_case += 1
            if asc_of_case == past_cases_interval - 1:
                asc_of_case = -1

        training_feature_matrix.loc[i - 31] = row
        row = []
    return training_feature_matrix

def predict_COVID_part1(svm_model, train_df, train_labels_df, past_cases_interval, past_weather_interval, test_feature):
    training_feature_matrix = construct_training_feature_matrix(past_cases_interval, past_weather_interval)
    training_feature_matrix = insert_value_into_training_feature_matrix(training_feature_matrix, past_weather_interval, past_cases_interval)
    # print(training_feature_matrix.values)
    X_train = training_feature_matrix.values
    y_train = train_labels_df.loc[30:, 'dailly_cases'].values.reshape(-1, 1).ravel()
    # print(X.shape)
    # print(y.shape)
    a = svm_model.fit(X_train, y_train)

    X_test_max_temp_series = test_feature[f"max_temp-{past_weather_interval}":"max_temp-1"]
    X_test_max_dew_series = test_feature[f"max_dew-{past_weather_interval}":"max_dew-1"]
    X_test_max_humid_series = test_feature[f"max_humid-{past_weather_interval}":"max_humid-1"]
    X_test_daily_cases_series = test_feature[f"dailly_cases-{past_cases_interval}":"dailly_cases-1"]
    # 把series转换为数据框，转置是为了后续合并后方便作为测试集
    X_test_max_temp = X_test_max_temp_series.to_frame().T
    X_test_max_dew = X_test_max_dew_series.to_frame().T
    X_test_max_humid = X_test_max_humid_series.to_frame().T
    X_test_daily_cases = X_test_daily_cases_series.to_frame().T
    X_test = ((X_test_max_temp.join(X_test_max_dew)).join(X_test_max_humid)).join(X_test_daily_cases)
    # print(X_test)
    prediction = a.predict(X_test)
    return floor(prediction[0])
    # print(floor(b[0]))

    # training_feature_matrix.to_csv(path_or_buf="matrix.csv", index=False)

## Project-Part2
def predict_COVID_part2(train_df, train_labels_df, test_feature):
    pass ## Replace this line with your implementation




## Parameters settings
past_cases_interval = 10
past_weather_interval = 10
## Read training data
train_file = 'COVID_train_data.csv'
train_df = pd.read_csv(train_file)
## Read Training labels
train_label_file = 'COVID_train_labels.csv'
train_labels_df = pd.read_csv(train_label_file)
## Read testing Features
test_fea_file = 'test_features.csv'
test_features = pd.read_csv(test_fea_file)
## Set hyper-parameters for the SVM Model
svm_model = SVR()
svm_model.set_params(**{'kernel': 'rbf', 'degree': 1, 'C': 5000,
                       'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})
## Generate Prediction Results
predicted_cases_part1 = []
for idx in range(len(test_features)):
   test_feature = test_features.loc[idx]
   # print(test_feature)
   prediction = predict_COVID_part1(svm_model, train_df, train_labels_df,
                                               past_cases_interval, past_weather_interval, test_feature)
   predicted_cases_part1.append(prediction)
print(predicted_cases_part1)