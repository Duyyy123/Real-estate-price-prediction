import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load DataFrame
df = pd.read_csv("real_estate.csv")
columns_multiindex = pd.MultiIndex.from_tuples([(col, i) for i, col in enumerate(df.columns)])
df.columns = columns_multiindex

# feature extraction and labeling
data = df.drop([0, 7], axis = 1, level = 1)
label = df["Y house price of unit area"]

# data splitting
train_data, train_label, test_data, test_label = train_test_split(data, label, test_size = 0.1, random_state = 50)

# data normalization
test_data = test_data.to_numpy()
train_data = train_data.to_numpy()

test_label = test_label.to_numpy()
train_label = train_label.to_numpy()

# show data
print("train data shape: ", train_data.shape)
print("train label shape: ", train_label.shape)
print("test data shape: ", test_data.shape)
print("test label shape: ", test_label.shape)