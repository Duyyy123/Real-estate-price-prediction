import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
df = pd.read_csv("real_estate.csv")

data = df.drop('Y', axis = 1)
label = df['Y']

train_data, test_data, train_label, test_label = train_test_split(data, label, test_size = 0.1, random_state = 50); 

test_data = test_data.to_numpy()
train_data = train_data.to_numpy()

test_label = test_label.to_numpy()
train_label = train_label.to_numpy()

print("Chieu du lieu train", train_data.shape)
print("Chieu nhan train", train_label.shape)

print()

print("Chieu du lieu test", test_data.shape)
print("Chieu nhan test", test_label.shape)