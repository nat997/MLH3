import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
print("Median age ",titanic_data_train['Age'].mean())
print("Median age ",titanic_data_train['Age'].median())