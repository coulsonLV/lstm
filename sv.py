import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_csv('processed_data/i-80.1sv.csv')
data = data.sort_values(by = 'frame')
data_value = data.values
sv = [] 

for i in tqdm(range(len(data)), ascii=True):
    frame = data_value[i][2]
    x_pos = data_value[i][3]
    y_pos = data_value[i][4]
    lane = data_value[i][8]

    left_front = data.loc[(data['frame']==frame) & (data['y_position']-y_pos<20) & (data['y_position']-y_pos>0) & (data['lane']-lane==1),
                            ['velocity','a']]
    front = data.loc[(data['frame']==frame) & (data['y_position']-y_pos<15) & (data['y_position']-y_pos>0) & (data['lane']==lane),
                            ['velocity','a']]
    right_front = data.loc[(data['frame']==frame) & (data['y_position']-y_pos<20) & (data['y_position']-y_pos>0) & (lane-data['lane']==1),
                            ['velocity','a']]
    left_behind = data.loc[(data['frame']==frame) & (y_pos-data['y_position']<20) & (y_pos-data['y_position']>0) & (data['lane']-lane==1),
                            ['velocity','a']]
    behind = data.loc[(data['frame']==frame) & (y_pos-data['y_position']<15) & (y_pos-data['y_position']>0) & (data['lane']==lane),
                            ['velocity','a']]
    right_behind = data.loc[(data['frame']==frame) & (y_pos-data['y_position']<20) & (y_pos-data['y_position']>0) & (lane-data['lane']==1),
                            ['velocity','a']]
    
   
    if left_front.empty:
        left_front = [1000,1000]
    else:
        left_front = [j for i in np.array(left_front) for j in i]
    if len(left_front) > 2:
        left_front = left_front[-2:]
    else:
        left_front = left_front
  
    if front.empty:
        front = [1000,1000]
    else:
        front = [j for i in np.array(front) for j in i]
    if len(front) > 2:
        front = front[-2:]
    else:
        front = front
      
    if right_front.empty:
        right_front = [1000,1000]
    else:
        right_front = [j for i in np.array(right_front) for j in i]
    if len(right_front) > 2:
        right_front = right_front[-2:]
    else:
        right_front = right_front
    
    if left_behind.empty:
        left_behind = [1000,1000]
    else:
        left_behind = [j for i in np.array(left_behind) for j in i]
    if len(left_behind) > 2:
        left_behind = left_behind[-2:]
    else:
        left_behind = left_behind

    if behind.empty:
        behind = [1000,1000]
    else:
        behind = [j for i in np.array(behind) for j in i]
    if len(behind) > 2:
        behind = behind[-2:]
    else:
        behind = behind

    if right_behind.empty:
        right_behind = [1000,1000]
    else:
        right_behind = [j for i in np.array(right_behind) for j in i]
    if len(right_behind) > 2:
        right_behind = right_behind[-2:]
    else:
        right_behind = right_behind

    vela = left_front + front + right_front + left_behind + behind + right_behind 
    sv.append(vela)

datav = data_value[:,6]
dataa = data_value[:,7]
number = data['number']
number = np.array(number).reshape(len(data),-1)
sv = np.array(sv)
lfv = sv[:,0]
lfa = sv[:,1]
fv = sv[:,2]
fa = sv[:,3]
rfv = sv[:,4]
rfa = sv[:,5]
lbv = sv[:,6]
lba = sv[:,7]
bv = sv[:,8]
ba = sv[:,9]
rbv = sv[:,10]
rba = sv[:,11]

lfv = lfv - datav
lfv[lfv>100] = 0
lfv = np.array(lfv).reshape(-1,1)
lfa = lfa - dataa
lfa[lfa>100] = 0
lfa = np.array(lfa).reshape(-1,1)
fv = fv - datav
fv[fv>100] = 0
fv = np.array(fv).reshape(-1,1)
fa = fa - dataa
fa[fa>100] = 0
fa = np.array(fa).reshape(-1,1)
rfv = rfv - datav
rfv[rfv>0] = 0
rfv = np.array(rfv).reshape(-1,1)
rfa = rfa - dataa
rfa[rfa>100] = 0
rfa = np.array(rfa).reshape(-1,1)
lbv = lbv - datav
lbv[lbv>100] = 0
lbv = np.array(lbv).reshape(-1,1)
lba = lba - dataa
lba[lba>0] = 0
lba = np.array(lba).reshape(-1,1)
bv = bv - datav 
bv[bv>100] = 0
bv = np.array(bv).reshape(-1,1)
ba = ba - dataa
ba[ba>0] = 0
ba = np.array(ba).reshape(-1,1)
rbv = rbv - datav
rbv[rbv>0] = 0
rbv = np.array(rbv).reshape(-1,1)
rba = rba - dataa
rba[rba>0] = 0
rba = np.array(rba).reshape(-1,1)
sv = np.hstack((lfv,lfa,fv,fa,rfv,rfa,lbv,lba,bv,ba,rbv,rba))
sv = np.hstack((number,sv))
sv = pd.DataFrame(sv,columns = ['number','v1','a1','v2','a2','v3','a3','v4','a4','v5','a5','v6','a6'])
final_data = pd.merge(data,sv,on = 'number')
final_data.to_csv('processed_data/i-80.1sv.csv') 

