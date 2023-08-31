#importer vos libs 
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
house_data = pd.read_csv("../data/house_pricing.csv", sep=",")
print(house_data.head())
print("Data loaded.")
columns_to_plot = ['bedrooms', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
pd.plotting.scatter_matrix(house_data[columns_to_plot], figsize=(12, 10), diagonal='hist')

# Display the plot
plt.show()