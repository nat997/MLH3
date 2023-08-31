import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import h2o
from sklearn.metrics import auc, roc_curve, classification_report
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')
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
predictors = X_train.columns.tolist()
response = 'class'
# Build Random Forest
drf = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
drf.train(x=predictors, y=response, training_frame=train)

# Extract feature importances from the model
feature_importances = drf.varimp(use_pandas=True)
feature_importances = feature_importances.sort_values(by='percentage', ascending=False)
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['variable'], feature_importances['percentage'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.savefig("Importances.png")

# classification report
y_test_pred = drf.predict(test).as_data_frame()
y_test_true = test['class'].as_data_frame()
y_test_pred['predict_class'] = y_test_pred['predict'].apply(lambda x: 1 if x > 0.5 else 0)
report = classification_report(y_test_true['class'], y_test_pred['predict_class'])



# Get predicted probabilities for positive class from H2O model

pred = drf.predict(test)
y_test_probs = pred.as_data_frame()['predict'].values

y_test_probs = drf.predict(test).as_data_frame()['predict']

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test_true['class'], y_test_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC_Fraud.png")

#print("AUC:", roc_auc)
#AUC: 0.8469369487371068





#print(report)
###result
"""
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     41117
           1       0.97      0.55      0.70      4217

    accuracy                           0.96     45334
   macro avg       0.97      0.77      0.84     45334
weighted avg       0.96      0.96      0.95     45334
"""


############################################
### Result
###print(drf)
"""Model Details
=============
H2ORandomForestEstimator : Distributed Random Forest
Model Key: DRF_model_python_1693404184912_1


Model Summary:
    number_of_trees    number_of_internal_trees    model_size_in_bytes    min_depth    max_depth    mean_depth    min_leaves    max_leaves    mean_leaves
--  -----------------  --------------------------  ---------------------  -----------  -----------  ------------  ------------  ------------  -------------
    50                 50                          2.28517e+06            20           20           20            2067          3648          2915.86

ModelMetricsRegression: drf
** Reported on train data. **

MSE: 0.04095810121937017
RMSE: 0.20238107920299805
MAE: 0.07992632740806771
RMSLE: 0.14310767319717757
Mean Residual Deviance: 0.04095810121937017

ModelMetricsRegression: drf
** Reported on cross-validation data. **

MSE: 0.04000380236158811
RMSE: 0.2000095056780755
MAE: 0.08015569808399489
RMSLE: 0.14070579404166939
Mean Residual Deviance: 0.04000380236158811

Cross-Validation Metrics Summary:
                        mean       sd          cv_1_valid    cv_2_valid    cv_3_valid    cv_4_valid    cv_5_valid    cv_6_valid    cv_7_valid    cv_8_valid    cv_9_valid    cv_10_valid
----------------------  ---------  ----------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  -------------
mae                     0.0801531  0.00133281  0.0780684     0.0826995     0.0795851     0.0798605     0.0802715     0.0819217     0.0789443     0.0799584     0.0798823     0.0803395
mean_residual_deviance  0.0400022  0.00138763  0.0383028     0.0429129     0.0388483     0.0401261     0.0404501     0.041476      0.0389366     0.0400453     0.0389476     0.0399767
mse                     0.0400022  0.00138763  0.0383028     0.0429129     0.0388483     0.0401261     0.0404501     0.041476      0.0389366     0.0400453     0.0389476     0.0399767
r2                      0.52977    0.0140406   0.530121      0.497613      0.546778      0.547068      0.520471      0.528445      0.53459       0.534866      0.526371      0.531373
residual_deviance       0.0400022  0.00138763  0.0383028     0.0429129     0.0388483     0.0401261     0.0404501     0.041476      0.0389366     0.0400453     0.0389476     0.0399767
rmse                    0.199979   0.00344654  0.195711      0.207154      0.1971        0.200315      0.201122      0.203657      0.197324      0.200113      0.197351      0.199942
rmsle                   0.140689   0.00213758  0.137929      0.145034      0.139054      0.140904      0.141161      0.143143      0.138937      0.140922      0.139096      0.140705

Scoring History:
     timestamp            duration    number_of_trees    training_rmse        training_mae         training_deviance
---  -------------------  ----------  -----------------  -------------------  -------------------  --------------------
     2023-08-30 16:03:54  44.770 sec  0.0                nan                  nan                  nan
     2023-08-30 16:03:55  44.948 sec  1.0                0.2388057471414161   0.07808040713911542  0.057028184867769977
     2023-08-30 16:03:55  45.033 sec  2.0                0.24332590534753898  0.07912563658306125  0.0592074962131995
     2023-08-30 16:03:55  45.099 sec  3.0                0.2371622817720719   0.07958601673456284  0.05624594789533563
     2023-08-30 16:03:55  45.162 sec  4.0                0.23396043983312606  0.07956384862854068  0.0547374874069098
     2023-08-30 16:03:55  45.218 sec  5.0                0.22995134259455557  0.07959089013192171  0.052877619961038666
     2023-08-30 16:03:55  45.254 sec  6.0                0.22682235114141683  0.079773760786633    0.0514483789773202
     2023-08-30 16:03:55  45.286 sec  7.0                0.22320459819000726  0.07953661360426657  0.049820292653162596
     2023-08-30 16:03:55  45.336 sec  8.0                0.22075884697712775  0.07952825546323175  0.0487344685186709
     2023-08-30 16:03:55  45.369 sec  9.0                0.21823581593110092  0.07944695430427666  0.04762687135511336
---  ---                  ---         ---                ---                  ---                  ---
     2023-08-30 16:03:56  46.506 sec  41.0               0.20301095852385767  0.07991798122349157  0.04121344928077546
     2023-08-30 16:03:56  46.541 sec  42.0               0.2029531332799979   0.07991602918626899  0.04118997430816859
     2023-08-30 16:03:56  46.579 sec  43.0               0.20287657413587423  0.07989419559111323  0.04115890433310888
     2023-08-30 16:03:56  46.612 sec  44.0               0.20277185440243375  0.07987675676218335  0.04111642493780179
     2023-08-30 16:03:56  46.645 sec  45.0               0.20272199053169496  0.07991207024867378  0.04109620544513262
     2023-08-30 16:03:56  46.678 sec  46.0               0.2026395312724718   0.07990443109195339  0.04106277963432708
     2023-08-30 16:03:56  46.710 sec  47.0               0.2025543246022876   0.07987893544550646  0.04102825441508889
     2023-08-30 16:03:56  46.748 sec  48.0               0.2025332340160395   0.07992919725346198  0.041019710880995826
     2023-08-30 16:03:56  46.782 sec  49.0               0.20245762736867054  0.07994082654989061  0.04098909087975146
     2023-08-30 16:03:56  46.819 sec  50.0               0.20238107920299805  0.07992632740806771  0.04095810121937017
[51 rows x 7 columns]


Variable Importances:
variable        relative_importance    scaled_importance    percentage
--------------  ---------------------  -------------------  ------------
time_diff       82128.6                1                    0.279951
ip_num          75039.1                0.913679             0.255785
device_num      37197.6                0.452919             0.126795
purchase_week   33734.9                0.410757             0.114992
country         14783.2                0.18                 0.0503912
signup_week     14700                  0.178988             0.0501077
purchase_value  8881.88                0.108146             0.0302756
age             8476.57                0.103211             0.028894
signup_day      4906.97                0.0597475            0.0167263
browser         4675.02                0.0569231            0.0159357
purchase_day    4515.58                0.0549818            0.0153922
source          2987.11                0.0363711            0.0101821
sex             1341.41                0.0163331            0.00457245
Closing connection _sid_a450 at exit
H2O session _sid_a450 closed.
"""
 