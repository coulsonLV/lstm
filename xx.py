import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from numpy import *
np.set_printoptions(threshold=2000)

data1 = pd.read_csv('data1/us-101.2sv.csv')

data1 = data1.groupby(["Vehicle_ID"])
data2 = data1.size().index
data = data1.size().values
a = (data-5)//35*35

print(a)

b = []
for i,j in data1:
    b.append(i)
b = np.array(b).reshape(1,-1)
print(b)

print(data2)
'''
# the first order Savitzky-Golay filter
data = data1.to_numpy()
data = data[:200,:]
x = data[:,1]
y = data[:,0]
plt.plot(x,y,color = 'red')

x1 = scipy.signal.savgol_filter(x,53,9)
y1 = scipy.signal.savgol_filter(y,53,9)

plt.plot(x1,y1,color='green')
plt.show()

z = [3.4,3.7,4.2,3.4,4.7,5.4]
z = np.array(z).reshape(-1,3)

x = [0,1,2]
x = np.array(x)
f=interp1d(x, z, kind='quadratic')
xp = np.linspace(x.min(), x.max(), 50)
zp=f(xp)
print(zp)


data1 = pd.read_csv('datafront/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv')
data1 = data1.to_numpy()
print(data1.shape)
'''