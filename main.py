import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress = True)
# load DataFrame
df = pd.read_csv("real_estate.csv", dtype = float)

# data normalization
cmin = df.max()
cmax = df.min()
df.iloc[:,:] = (df.iloc[:,:] - cmin[:]) / (cmax[:] - cmin[:])

# feature extraction and labeling
data = df.drop(["No", "Y house price of unit area"], axis = 1)
label = df[["Y house price of unit area"]]


# data splitting
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size = 0.1, random_state = 50)

# data pre-processing
test_data = test_data.to_numpy()
train_data = train_data.to_numpy()

test_label = test_label.to_numpy()
test_label = test_label.flatten()

train_label = train_label.to_numpy()
train_label = train_label.flatten()

# show data

# show feature and label
print("data: \n", data)
print("label: \n", label)
print()

# show train set
print("train data head: \n", train_data[:5, :])
print("train label head: \n", train_label[:5])

print("train data shape: ", train_data.shape)
print("train label shape: ", train_label.shape)
print()

# show test set
print("test data head : \n", test_data[:5, :])
print("test label head: \n", test_label[:5])

print("test data shape: ", test_data.shape)
print("test label shape: ", test_label.shape)
