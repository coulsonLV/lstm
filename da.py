import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_csv('processed_data/i-80.1sv.csv')
number = data['number']
number = np.array(number).reshape(-1,1)
data1 = data.to_numpy()
data1 = data1.astype('float32')
x = data1[:,6]
y = data1[:,5]
theta1 = []
for j in range(len(x)):
    if j < 1:
        theta1.append(0)
    else:
        deltax = x[j]-x[j-1]
        deltay = y[j]-y[j-1] 
        theta1.append(math.atan2(deltay, deltax))

theta1 = np.array(theta1).reshape(-1,1)
for j in range(len(theta1)):
    if abs(theta1[j])>0.5:
        theta1[j] = 0
theta1 = np.hstack((number,theta1))
theta1 = pd.DataFrame(theta1,columns=['number','theta1'])
data = pd.merge(data,theta1,on = 'number')

data.to_csv('processed_data/i-80.2sv.csv') 



