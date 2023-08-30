import warnings

from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve, classification_report
import matplotlib.pyplot as plt
import h2o
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator

data = pd.read_csv('../data/Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'])
#print(data.head())
address2country = pd.read_csv('../data/IpAddress_to_Country.csv')
#print(address2country.head())

#Because the order messed up so i gotta set up the order again cause some how it sorted order by the country Alphabetique so this is not asked in the website 
data['original_order'] = range(len(data))
# Merge the two datasets and print the first 5 rows
merged_data = pd.merge_asof(data.sort_values('ip_address'), 
                            address2country.sort_values('lower_bound_ip_address'), 
                            left_on='ip_address', 
                            right_on='lower_bound_ip_address', 
                            direction='forward', 
                            allow_exact_matches=True)
merged_data = merged_data.sort_values('original_order')
merged_data = merged_data.drop('original_order', axis=1)
#print(merged_data.head())

# Time diff
merged_data['time_diff'] = (merged_data['purchase_time'] - merged_data['signup_time']).dt.total_seconds() / 3600  # Time difference in hours
# Check user number for unique devices
merged_data['device_num'] = merged_data.groupby('device_id')['user_id'].transform('nunique')
# Check user number for unique ip_address
merged_data['ip_num'] = merged_data.groupby('ip_address')['user_id'].transform('nunique')
# Signup day and week
merged_data['signup_day'] = merged_data['signup_time'].dt.dayofweek  
merged_data['signup_week'] = merged_data['signup_time'].dt.isocalendar().week
# Purchase day and week
merged_data['purchase_day'] = merged_data['purchase_time'].dt.dayofweek 
merged_data['purchase_week'] = merged_data['purchase_time'].dt.isocalendar().week

#print(merged_data.head())

# Define features and target to be used
final_data = merged_data.drop(columns=['signup_time', 'purchase_time', 'device_id', 'ip_address', 'user_id'])
cols_order = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 'source', 'browser', 'sex', 'age', 'country', 'time_diff', 'device_num', 'ip_num', 'class']
final_data = final_data[cols_order]
#print(final_data.head())

# Define features and target
X = final_data.drop(columns=['class'])
y = final_data['class']

# Split into 70% training and 30% test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build random forest model
h2o.init()
# Convert pandas dataframes to H2OFrames
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
# Identify predictors and response
predictors = X_train.columns.tolist()
response = 'class'
# Build Random Forest Model
drf = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
drf.train(x=predictors, y=response, training_frame=train)



### LES OUTPUTS
""" 
On a les resultats 
Metrics on the training data:
MSE (Mean Squared Error): 0.0407
RMSE (Root Mean Squared Error): 0.2017
MAE (Mean Absolute Error): 0.0797
RMSLE (Root Mean Squared Logarithmic Error): 0.1425
Metrics on the cross-validation data:
MSE: 0.0399
RMSE: 0.1999
MAE: 0.0800
RMSLE: 0.1406 
"""

"""
variable        relative_importance    scaled_importance    percentage
--------------  ---------------------  -------------------  ------------
time_diff       85364.8                1                    0.287509
ip_num          66359.8                0.777368             0.2235
device_num      45126                  0.528625             0.151984
purchase_week   41168.8                0.482269             0.138657
country         14273.1                0.167202             0.0480719
signup_week     10437.2                0.122266             0.0351525
purchase_value  8502.5                 0.0996019            0.0286364
age             8042.55                0.094214             0.0270873
signup_day      4586.36                0.0537266            0.0154469
purchase_day    4516.52                0.0529085            0.0152117
browser         4347.05                0.0509232            0.0146409
source          2667.34                0.0312463            0.00898359
sex             1520                   0.017806             0.00511937
"""

"""
8 Features Matrix:
[[89 16]
 [19 55]]
              precision    recall  f1-score   support

           0       0.82      0.85      0.84       105
           1       0.77      0.74      0.76        74

    accuracy                           0.80       179
   macro avg       0.80      0.80      0.80       179
weighted avg       0.80      0.80      0.80       179
"""
# Variables and their scaled importances
variables = ["time_diff", "ip_num", "device_num", "purchase_week", "country", 
            "signup_week", "purchase_value", "age", "signup_day", "purchase_day"]
scaled_importance = [1, 0.777368, 0.528625, 0.482269, 0.167202, 
                    0.122266, 0.0996019, 0.094214, 0.0537266, 0.0529085]
plt.figure(figsize=(10,6))
plt.barh(variables, scaled_importance, color='dodgerblue')
plt.xlabel('Scaled Importance')
plt.ylabel('Variable')
plt.title('Variable Importances')
plt.gca().invert_yaxis() 
plt.savefig("Importances.png")

#plot ROC curve and calculate AUC
performance = drf.model_performance(test)
roc = performance.roc()
fpr = roc[0]
tpr = roc[1]
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
auc_score = performance.auc()
print(f"AUC: {auc_score:.4f}")
plt.savefig("ROCFraud.png")