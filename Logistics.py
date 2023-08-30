import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, log_loss, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.feature_selection import RFE, RFECV
# Load the dataset
titanic_data_test = pd.read_csv("../data/test.csv",sep=",")
titanic_data_train = pd.read_csv("../data/train.csv",sep=",")
titanic_data_gender_submission = pd.read_csv("../data/gender_submission.csv",sep=",")
taille_train = len(titanic_data_train)
taille_test = len(titanic_data_test)
print("Data loaded.")
print(titanic_data_test.head())
# Print dataset shape
print("Taille du dataset d'entrainement :", taille_train)
print("Taille du dataset de test :", taille_test)
#combined_data = pd.concat([titanic_data_train, titanic_data_test])
# Create a heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(titanic_data_train.isnull(), cmap="viridis")
plt.title("Missing Data: NaN Values Heatmap")
plt.savefig("logictics_heatmap.png")
# Calculate the percentage by doing the total of the age then divide by its numbers
age_nan_percentage = (titanic_data_train['Age'].isnull().sum() / len(titanic_data_train['Age'])) * 100
print(f"Percentage of NaN values in the 'Age' column: {age_nan_percentage:.2f}%")
# Print histogram of the 'Age' column
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data_train, x='Age', bins=30, kde=True)
plt.title("Histogram of Age Column")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_histogram.png")
# Print the median et moyenne
print("Mean age ",titanic_data_train['Age'].mean())
print("Median age ",titanic_data_train['Age'].median())
# percentage of NaN
cabin_nan_percentage = (titanic_data_train['Cabin'].isnull().sum() / len(titanic_data_train['Cabin'])) * 100
print(f"Percentage NaN value 'Cabin' column: {cabin_nan_percentage:.2f}%")

#'Embarked' column
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data_train, x='Embarked')
plt.title("Distribution of Embarked Column")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.savefig("embarked_distribution.png")

# If "Age" is missing for a given row, we assign 28 (median age).
titanic_data_train['Age'].fillna(titanic_data_train['Age'].median(), inplace=True)

# If "Embarked" is missing for a given line, we assign "S" (most common embarked).
most_common_embarked = titanic_data_train['Embarked'].value_counts().idxmax()
titanic_data_train['Embarked'].fillna(most_common_embarked, inplace=True)

#Check result 
missing_values = titanic_data_train.isnull().sum()
print(missing_values)

#create the feature 'TravelAlone'
titanic_data_train['TravelAlone'] = (titanic_data_train['SibSp'] + titanic_data_train['Parch'] == 0).astype(int)
print(titanic_data_train.head())

#use the get_dummies function to encode every needed features
categorical_features = ['Pclass', 'Embarked', 'Sex']
titanic_encoded = pd.get_dummies(titanic_data_train, columns=categorical_features, drop_first=False)
print(titanic_encoded.head())

#drop the columns that you don't need
columns_to_drop = ['PassengerId', 'Name', 'Ticket']
titanic_filtered = titanic_encoded.drop(columns=columns_to_drop)
print(titanic_filtered.head())

#Do the same with test dataset
titanic_data_test['Age'].fillna(titanic_data_test['Age'].median(), inplace=True)
most_common_embarked_test = titanic_data_test['Embarked'].value_counts().idxmax()
titanic_data_test['Embarked'].fillna(most_common_embarked_test, inplace=True)
titanic_data_test['Fare'].fillna(titanic_data_test['Fare'].median(), inplace=True)
titanic_data_test['TravelAlone'] = (titanic_data_test['SibSp'] + titanic_data_test['Parch'] == 0).astype(int)
categorical_features = ['Pclass', 'Embarked', 'Sex']
titanic_encoded_test = pd.get_dummies(titanic_data_test, columns=categorical_features, drop_first=False)
columns_to_drop_test = ['PassengerId', 'Name', 'Ticket']
titanic_filtered_test = titanic_encoded_test.drop(columns=columns_to_drop_test)
missing_values_test = titanic_filtered_test.isnull().sum()
print("From DATA TEST : ")
print(missing_values_test)
print(titanic_filtered.head())


#print the age feature distribution and highlight the survived feature
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=titanic_filtered, x='Age', hue='Survived', fill=True, common_norm=False)
plt.title("Distribution of Age with Survival")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.savefig("age_feature_distributio.png")

#add the "IsMinor" variable to your data (a person is considered a minor if they are under 16)
titanic_filtered['IsMinor'] = (titanic_filtered['Age'] < 16).astype(int)
print(titanic_filtered.head())
#fonction pour les plots
def plot_survivors(data, feature_columns, feature_labels, title, filename, palette=None):
    survivor_rates = [data[data[col] == 1]['Survived'].mean() for col in feature_columns]
    sns.barplot(x=feature_labels, y=survivor_rates, palette=palette)
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Survivors rate')
    plt.savefig(filename)

# for Pclass
feature_columns_class = ['Pclass_1', 'Pclass_2', 'Pclass_3']
feature_labels_class = ['Class 1', 'Class 2', 'Class 3']
plot_survivors(titanic_filtered, feature_columns_class, feature_labels_class, 
              'Number of Survivors by Class', "survivors_by_pclass.png")

# for Embarked
feature_columns_embarked = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
feature_labels_embarked = ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)']
plot_survivors(titanic_filtered, feature_columns_embarked, feature_labels_embarked, 
              'Number of Survivors by Embarkation Point', "survivors_by_embarked.png", "viridis")

# for Traveling Alone feature
travel_alone_survivor_rate = titanic_filtered[titanic_filtered['TravelAlone'] == 0]['Survived'].mean()
not_travel_alone_survivor_rate = 1 - travel_alone_survivor_rate
feature_labels_alone = ['Not Traveling Alone', 'Traveling Alone']
sns.barplot(x=feature_labels_alone, y=[not_travel_alone_survivor_rate, travel_alone_survivor_rate])
plt.title('Survivor Rate by Traveling Status')
plt.xlabel('Traveling Status')
plt.ylabel('Survivor Rate')
plt.savefig("survivors_by_travel_alone.png")


# for Gender distribution of survivors
female_survivor_rate = titanic_filtered[titanic_filtered['Sex_female'] == 1]['Survived'].mean()
male_survivor_rate = titanic_filtered[titanic_filtered['Sex_male'] == 1]['Survived'].mean()
feature_labels_gender = ['Female', 'Male']
sns.barplot(x=feature_labels_gender, y=[female_survivor_rate, male_survivor_rate], palette="pastel")
plt.title('Survivor Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survivor Rate')
plt.savefig("survivors_by_gender.png")


#################
X = titanic_filtered.drop(['Survived'], axis=1)
y = titanic_filtered["Survived"]
logisticreg = LogisticRegression(max_iter=1000) 
X_4_features = titanic_filtered[['Pclass_1', 'Pclass_2', 'Sex_male', 'IsMinor']]
X_8_features = titanic_filtered[['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 'Embarked_S', 'Sex_male', 'IsMinor']]
logisticreg.fit(X_4_features, y)
y_pred_4 = logisticreg.predict(X_4_features)
logisticreg.fit(X_8_features, y)
y_pred_8 = logisticreg.predict(X_8_features)

# Split data 
X4_train, X4_test, y4_train, y4_test = train_test_split(X_4_features, y, test_size=0.2, random_state=42)
X8_train, X8_test, y8_train, y8_test = train_test_split(X_8_features, y, test_size=0.2, random_state=42)

# Train model on 4 features
logisticreg.fit(X4_train, y4_train)
# Predict for 4 features
y4_pred = logisticreg.predict(X4_test)
y4_pred_proba = logisticreg.predict_proba(X4_test)[:, 1]

# Train model on 8 features
logisticreg.fit(X8_train, y8_train)
# Predict for 8 features
y8_pred = logisticreg.predict(X8_test)
y8_pred_proba = logisticreg.predict_proba(X8_test)[:, 1]

### https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

# Calculate the ROC curve for 4 features
fpr_4, tpr_4, thresholds_4 = roc_curve(y4_test, y4_pred_proba)
roc_auc_4 = auc(fpr_4, tpr_4)

# Calculate the ROC curve for 8 features
fpr_8, tpr_8, thresholds_8 = roc_curve(y8_test, y8_pred_proba)
roc_auc_8 = auc(fpr_8, tpr_8)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(fpr_4, tpr_4, color='blue', label='ROC curve for 4 features (area = %0.2f)' % roc_auc_4)
plt.plot(fpr_8, tpr_8, color='green', label='ROC curve for 8 features (area = %0.2f)' % roc_auc_8)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("ROC.png")

# Evaluate 
print("4 Features Model :")
print("Accuracy:", accuracy_score(y4_test, y4_pred))
print("Log Loss:", log_loss(y4_test, y4_pred_proba))
print("AUC:", roc_auc_score(y4_test, y4_pred_proba))
print("\n8 Features Model :")
print("Accuracy:", accuracy_score(y8_test, y8_pred))
print("Log Loss:", log_loss(y8_test, y8_pred_proba))
print("AUC:", roc_auc_score(y8_test, y8_pred_proba))
