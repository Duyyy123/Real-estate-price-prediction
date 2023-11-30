import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing

np.set_printoptions(suppress = True)
# load DataFrame
df = pd.read_csv("real_estate.csv", dtype = float)

# data normalization
cmin = df.max()
cmax = df.min()
df.iloc[:,:] = (df.iloc[:,:] - cmin[:]) / (cmax[:] - cmin[:])

# feature extraction and labeling
data = df[:, 1:7]
label = df[:, 7].reshape(-1)

# data splitting
train_data, test_data, train_label, test_label = model_selection.train_test_split(data, label, test_size = 0.1, random_state = 50)

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
