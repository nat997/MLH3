import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
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

def plot_survivors(data, feature_columns, feature_labels, title, filename, palette=None):
    survivors = [data[data[col] == 1]['Survived'].sum() for col in feature_columns]

    sns.barplot(x=feature_labels, y=survivors, palette=palette)
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Number of Survivors')
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
feature_labels_alone = ['Not Traveling Alone', 'Traveling Alone']
travel_alone_survivors = titanic_filtered['Survived'].sum() - titanic_filtered[titanic_filtered['TravelAlone'] == 0]['Survived'].sum()
not_travel_alone_survivors = titanic_filtered[titanic_filtered['TravelAlone'] == 0]['Survived'].sum()
sns.barplot(x=feature_labels_alone, y=[not_travel_alone_survivors, travel_alone_survivors])
plt.title('Number of Survivors by Traveling Status')
plt.xlabel('Traveling Status')
plt.ylabel('Number of Survivors')
plt.savefig("survivors_by_travel_alone.png")


# for Gender distribution of survivors
feature_labels_gender = ['Female', 'Male']
female_survivors = titanic_filtered[titanic_filtered['Sex_female'] == 1]['Survived'].sum()
male_survivors = titanic_filtered[titanic_filtered['Sex_male'] == 1]['Survived'].sum()
sns.barplot(x=feature_labels_gender, y=[female_survivors, male_survivors], palette="pastel")
plt.title('Number of Survivors by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Survivors')
plt.savefig("survivors_by_gender.png")


#################
# Assuming titanic_filtered is your DataFrame
X = titanic_filtered.drop(['Survived', 'Cabin'], axis=1)
y = titanic_filtered["Survived"]
logisticreg = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
X_4_features = titanic_filtered[['Pclass_1', 'Pclass_2', 'Sex_male', 'IsMinor']]
X_8_features = titanic_filtered[['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 'Embarked_S', 'Sex_male', 'IsMinor']]

# Model with 4 features
logisticreg.fit(X_4_features, y)
y_pred_4 = logisticreg.predict(X_4_features)

# Model with 8 features
logisticreg.fit(X_8_features, y)
y_pred_8 = logisticreg.predict(X_8_features)