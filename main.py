import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing

np.set_printoptions(suppress = True)
# load DataFrame
df = pd.read_csv("real_estate.csv", dtype = float)

# data normalization
df = preprocessing.MinMaxScaler().fit_transform(df)

# extraction feature and label
data = df[:, 1:7]
label = df[:, 7].reshape(-1)

# data splitting
train_data, test_data, train_label, test_label = model_selection.train_test_split(data, label, test_size = 0.1, random_state = 50)

#Set values
alpha = 0.00075
epsilon = 1e-6
max_loops = 5000

#random w, b
w = np.random.rand(1, 6)
b = np.random.rand(1, 1)

#compute L
def compute(w, b):
    N = len(train_data)
    L = 0.0
    for i in range (N):
        L += 0.5 * ((w@train_data[i] + b - train_label[i]) ** 2)
    return L

#derivative
def derivative(w, b):
    N = len(train_data)
    dw = np.zeros_like(w)
    db = 0.0
    for i in range (N):
        dw += (w @ train_data[i] + b - train_label[i]) * train_data[i]
        db += w @ train_data[i] + b - train_label[i]
    return dw, db

#gradient_descent
k = 0
pre_L = float('inf')
while k < max_loops:
    dw, db = derivative(w,b)
    w = w - alpha * dw;
    b = b - alpha * db;
    
    L = compute(w,b)
    
    if abs(L - pre_L) < epsilon:
        break
    
    pre_L = L
    k += 1
    

#TEST R^2
test_pre = (test_data @ w.T) + b
N = len(test_data)
tu = mau = 0
for i in range(N):
    tu += (test_label[i] - test_pre[i]) ** 2;
    mau += (test_label[i] - np.mean(test_label)) ** 2

test_r2 = 1 - (tu/mau)
    
print(f'Test R^2: {test_r2}')
