import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load DataFrame
df = pd.read_csv("real_estate.csv")

# feature extraction and labeling
data = df.drop(["No", "Y house price of unit area"], axis = 1)
label = df[["Y house price of unit area"]]

# data splitting
train_data, train_label, test_data, test_label = train_test_split(data, label, test_size = 0.1, random_state = 50)

# data normalization
test_data = test_data.to_numpy()
train_data = train_data.to_numpy()

test_label = test_label.to_numpy()
train_label = train_label.to_numpy()

# show data
print("features: \n", type(data))
print("label: \n", type(label))
print("train data shape: ", train_data.shape)
print("train label shape: ", train_label.shape)
print("test data shape: ", test_data.shape)
print("test label shape: ", test_label.shape)
