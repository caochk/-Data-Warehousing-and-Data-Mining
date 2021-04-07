import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from math import floor
from sklearn import preprocessing



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

def insert_value_into_training_feature_matrix(training_feature_matrix, past_weather_interval, past_cases_interval, train_df):
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
    training_feature_matrix = insert_value_into_training_feature_matrix(training_feature_matrix, past_weather_interval, past_cases_interval, train_df)
    # print(training_feature_matrix.values)
    X_train = training_feature_matrix.values
    y_train = train_labels_df.loc[30:, 'dailly_cases'].values.reshape(-1, 1).ravel()
    # print(X.shape)
    # print(y.shape)
    model_fit = svm_model.fit(X_train, y_train)

    # 构建测试集
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
    prediction = model_fit.predict(X_test)

    # plt.scatter(X_train, y_train, c='k', label='data', zorder=1)
    # plt.plot(X_test, prediction, c='r', label='SVR_fit')
    # plt.xlabel('data')
    # plt.ylabel('target')
    # plt.title('SVR versus Kernel Ridge')
    # plt.legend()
    # plt.show()
    return floor(prediction[0])

## Project-Part2
def construct_training_feature_matrix_part2():
    past_cases_interval = 30
    past_weather_interval = 30
    columns = []
    weather_features = ['max_temp', 'max_humid', 'max_wind_speed', 'precipitation'] #【机动变化】若后续需要加入avg及min，首先在此处加入
    for weather_feature in weather_features:
        for j in range(past_weather_interval):
            feature = f'{weather_feature}-{past_weather_interval - j}'
            columns.append(feature)
    for i in range(past_cases_interval):
        feature = f'past_cases-{past_cases_interval - i}'
        columns.append(feature)
    training_feature_matrix = pd.DataFrame(columns=columns)  # 一个空的数据框，之后会往其中加入数据以构建训练集
    return training_feature_matrix

def insert_value_into_training_feature_matrix_part2(training_feature_matrix, train_df):
    past_weather_interval = 30
    past_cases_interval = 30
    num_of_columns_of_matrix = past_weather_interval * 4 + past_cases_interval #【机动变化】若后续加入了avg,min，数字4需变
    row = []
    asc_of_weather = -1
    asc_of_case = -1
    for i in range(31, 193):
        for j in range(num_of_columns_of_matrix):
            if j < past_weather_interval: #【机动变化】若加入了avg,min，此处的判断语句需要加入avg及min的判断
                column_name = 'max_temp'
            elif j >= past_weather_interval and j < 2 * past_weather_interval:
                column_name = 'max_humid'
            elif j >= 2 * past_weather_interval and j < 3 * past_weather_interval:
                column_name = 'max_wind_speed'
            elif j >= 3 * past_weather_interval and j < 4 * past_weather_interval:
                column_name = 'precipitation'
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

def predict_COVID_part2(train_df, train_labels_df, test_feature):
    past_weather_interval = 30 # 【可能机动变换】若不打算取30天的数据时需更改
    past_cases_interval = 30 #【可能机动变换】同上
    training_feature_matrix = construct_training_feature_matrix_part2()
    training_feature_matrix = insert_value_into_training_feature_matrix_part2(training_feature_matrix, train_df)
    # training_feature_matrix.to_csv(path_or_buf="matrix_part2.csv", index=False)
    X_train = training_feature_matrix.values
    y_train = train_labels_df.loc[30:, 'dailly_cases'].values.reshape(-1, 1).ravel()
    X_train_scale = preprocessing.scale(X_train) # 还不太理解，失败！
    y_train_scale = preprocessing.scale(y_train) # 还不太理解，失败！
    # print(X_train_scale.mean(axis=0))

    svm_model = SVR()
    svm_model.set_params(**{'kernel': 'rbf', 'degree': 1, 'C': 5000,
                           'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})
    model_fit = svm_model.fit(X_train_scale, y_train_scale)
    # 构建测试集【机动变换】若后续加入了avg,min，下列所有行都需要再加入相应代码行
    X_test_max_temp_series = test_feature[f"max_temp-{past_weather_interval}":"max_temp-1"]
    X_test_max_humid_series = test_feature[f"max_humid-{past_weather_interval}":"max_humid-1"]
    X_test_max_wind_speed_series = test_feature[f"max_wind_speed-{past_weather_interval}":"max_wind_speed-1"]
    X_test_precipitation_series = test_feature[f"precipitation-{past_weather_interval}":"precipitation-1"]
    X_test_daily_cases_series = test_feature[f"dailly_cases-{past_cases_interval}":"dailly_cases-1"]
    # 把series转换为数据框，转置是为了后续合并后方便作为测试集
    X_test_max_temp = X_test_max_temp_series.to_frame().T
    X_test_max_humid = X_test_max_humid_series.to_frame().T
    X_test_max_wind_speed = X_test_max_wind_speed_series.to_frame().T
    X_test_precipitation = X_test_precipitation_series.to_frame().T
    X_test_daily_cases = X_test_daily_cases_series.to_frame().T
    X_test = (((X_test_max_temp.join(X_test_max_humid)).join(X_test_max_wind_speed)).join(X_test_precipitation)).join(X_test_daily_cases)
    prediction = model_fit.predict(X_test)
    return floor(prediction[0])



# ## Parameters settings
# past_cases_interval = 10
# past_weather_interval = 10
# ## Read training data
# train_file = 'COVID_train_data.csv'
# train_df = pd.read_csv(train_file)
# ## Read Training labels
# train_label_file = 'COVID_train_labels.csv'
# train_labels_df = pd.read_csv(train_label_file)
# ## Read testing Features
# test_fea_file = 'test_features.csv'
# test_features = pd.read_csv(test_fea_file)
# ## Set hyper-parameters for the SVM Model
# svm_model = SVR()
# svm_model.set_params(**{'kernel': 'rbf', 'degree': 1, 'C': 145000,
#                        'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 10})
# ## Generate Prediction Results
# predicted_cases_part1 = []
# for idx in range(len(test_features)):
#    test_feature = test_features.loc[idx]
#    # print(test_feature)
#    prediction = predict_COVID_part1(svm_model, train_df, train_labels_df,
#                                                past_cases_interval, past_weather_interval, test_feature)
#    predicted_cases_part1.append(prediction)
# # print(predicted_cases_part1)




## Read training data
train_file = 'COVID_train_data.csv'
train_df = pd.read_csv(train_file)
## Read Training labels
train_label_file = 'COVID_train_labels.csv'
train_labels_df = pd.read_csv(train_label_file)
## Read testing Features
test_fea_file = 'test_features.csv'
test_features = pd.read_csv(test_fea_file)
## Generate Prediction Results
predicted_cases_part2 = []
for idx in range(len(test_features)):
   test_feature = test_features.loc[idx]
   prediction = predict_COVID_part2(train_df, train_labels_df, test_feature)
   predicted_cases_part2.append(prediction)
print(predicted_cases_part2)


## MeanAbsoluteError Computation...!
test_label_file ='COVID_test_labels.csv'
test_labels_df = pd.read_csv(test_label_file)
ground_truth = test_labels_df['dailly_cases'].to_list()
MeanAbsError = mean_absolute_error(predicted_cases_part2, ground_truth)
print('MeanAbsError = ', MeanAbsError)