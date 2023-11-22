import pandas as pd
import numpy as np
 
df = pd.read_csv("real_estate.csv")

test_data = df.iloc[:41,:6]
train_data = df.iloc[41:,:6]

test_data = test_data.to_numpy()
train_data = train_data.to_numpy()

test_label = df.iloc[:41,7:]
train_label = df.iloc[41:,7:]

test_label = test_label.to_numpy()
train_label = train_label.to_numpy()

print("Chieu du lieu train", train_data.shape)
print("Chieu nhan train", train_label.shape)
print()
print("Chieu du lieu test", test_data.shape)
print("Chieu nhan test", test_label.shape)