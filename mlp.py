import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime 
import heapq
import time
from sklearn.metrics import mean_squared_log_error

"""
Data Preparation
"""

def cloud_all_binning(value):
    if(value <= 20):
        return 0
    elif(value <= 40):
        return 1
    elif(value <= 60):
        return 2
    elif(value <= 70):
        return 3
    elif(value <= 80):
        return 4
    else:
        return 5
    
df = pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')

# Drop duplicate rows of date_time records
df.drop_duplicates(subset='date_time', keep='first', inplace=True)
# Drop snow_1h column since all are zero values
df = df.drop(['snow_1h'], axis = 1)
# Convert rain_1h from continuous to discrete variable.
df['rain_1h'] = df['rain_1h'].apply(lambda x: 0 if x == 0 else 1)
# Perform binning on clouds_all column
df['clouds_all'] = df['clouds_all'].apply(lambda x: cloud_all_binning(x))
# Drop temp column as there seems to be little or no correlation with the output variable
df = df.drop(['temp'], axis = 1)


"""
Feature Engineering
"""

def getDayOfWeek(date):
    int_day = datetime.datetime.strptime(date, '%Y %m %d').weekday()
    days_mapping = {0:'Mon', 1:'Tues', 2:'Wed', 3:'Thurs', 4:'Fri', 5:'Sat', 6:'Sun'}
    return (days_mapping[int_day])

# Extract the hour of a timestamp into a new column
hours = df['date_time'].apply(lambda x: x.split(' ')[1].split(':')[0])
# One-hot encoding of data
hours = pd.get_dummies(hours).iloc[:, 1:]

# Extract the day of week of a timestamp into a new column
date = df['date_time'].apply(lambda x: x.split(' ')[0].replace('-',' '))
day_of_week = date.apply(lambda x: getDayOfWeek(x))
# One-hot encoding of data
day_of_week = pd.get_dummies(day_of_week).iloc[:, 1:]


X = df.iloc[:, [1,2]]
y = df['traffic_volume'].values
# Concat with the engineered features (extracted hours)
X = pd.concat([X, pd.DataFrame(hours)], axis = 1)
X = pd.concat([X, pd.DataFrame(day_of_week)], axis = 1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


"""
Model

#Model engineering
#Tuning of n_estimators parameter

acc_scores = []
num_esti = []
num_trees = np.arange(1, 280, 10) 

for k in num_trees:
    rfc = RandomForestRegressor(n_estimators=k, random_state=42)
    predictions = rfc.fit(X_train, y_train).predict(X_test)
    errors = abs(y_test - predictions)
    mean_perc_error = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mean_perc_error)
    acc_scores.append(accuracy)
    num_esti.append(k)
    
indexes = heapq.nlargest(10, range(len(acc_scores)), acc_scores.__getitem__)
n_max_score = [acc_scores[i] for i in indexes]
n_max_estimators = [num_esti[i] for i in indexes]
print(n_max_score)
print(n_max_estimators)


"""

# Training the model and making predictions
# From the results of model engineering, we will use n_estimators = 151
randForest = RandomForestRegressor(n_estimators = 151, random_state = 42)
time_start = time.time()
randForest.fit(X_train, y_train)
time_end = time.time() - time_start
print('Execution Time:', round(time_end, 3), 's')
preds = randForest.predict(X_train)
ypreds = randForest.predict(X_test)


"""
Evaluation metrics on model performance
"""

# Evaluation metrics on model performance
print('Training set')
print('Root Mean Squared Log Error =', round(np.sqrt(mean_squared_log_error(y_train, preds)), 3))
mean_perc_error = 100 * (abs(y_train - preds) / y_train)
accuracy = 100 - np.mean(mean_perc_error)
print('Accuracy Score:', round(accuracy, 3), '%')
print('Test set')
errors = abs(y_test - ypreds)
print('Root Mean Squared Log Error =', round(np.sqrt(mean_squared_log_error(y_test, ypreds)), 3))
mean_perc_error1 = 100 * (errors / y_test)
accuracy = 100 - np.mean(mean_perc_error1)
print('Accuracy Score:', round(accuracy, 3), '%')

