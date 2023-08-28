import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model #modèle linéaire
from sklearn.metrics import mean_squared_error, r2_score #métriques d'évaluation
from sklearn.model_selection import train_test_split

prices = pd.read_csv("../data/price_availability.csv", sep=";")
listings = pd.read_csv("../data/listings_final.csv", sep=";")
listings = listings.drop(589)  
print("Data loaded.")
#define our input variable X and output variable Y
X = listings.loc[:, ["listing_id", "person_capacity", "bedrooms", "bathrooms" ]]
Y = []
#build the price vector
for i, row in X.iterrows():
    y = 0
    ID = int(row["listing_id"])
    subset = prices[prices["listing_id"] == ID]
    y = subset["local_price"].mean()
    Y.append(y)

#convert into numpy array
Y = np.asarray(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

regr = linear_model.LinearRegression()
regr.fit(X_train.values, y_train)

#what do you think about the results ?
print('Coefficients beta_j : \n', regr.coef_)
print('Coefficients INTERCEPT beta_0 : \n', regr.intercept_)

X_test
y_test

#compute y_pred
Y_pred = regr.predict(X_test)
len(Y_pred)

Y_pred

y_test

#afficher l'erreur des moindres carrées sur l'ensemble d'entrainement ainsi que le R2
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, Y_pred))
# Coefficient de détermination R2
print('Variance score: %.2f' % r2_score(y_test, Y_pred))

#compute the RMSE for more intuitive results
np.sqrt(19631.83)