import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# find mode of total frames for given vehicle ID


def find_mode(data, Id):
    # np.where() has extremely low computational efficiency
    # d = data[np.where([data[i, 0] == Id for i in range(len(data))])]
    d = data[data[:, 0] == Id]
    d_frame = d[:, 2]
    count = np.bincount(d_frame.astype(int))
    return np.argmax(count)


''' filter data with the following columns only and re-sort data'''
# data = pd.read_csv('data/test.csv')
data = pd.read_csv(
    'data/trajectories-0820am-0835am.csv')
print(list(data))
# data = pd.read_csv('data/i-80_lane12345.csv')
data = data[['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Local_X',
             'Local_Y', 'v_Vel', 'Lane_ID', 'v_Length', 'v_Width', 'v_Class']].sort_values(by=['Vehicle_ID', 'Frame_ID'])
'''
data['Local_Y'] = [x.replace(',', '')
                   for x in data['Local_Y']]   # remove ',' in Local_Y
# data['Total_Frames'] = [x.replace(',', '')
#                         for x in data['Total_Frames']]   # remove ',' in Local_Y
'''
# add lane change label
data['label'] = None
data = data.astype(np.float32)
data = data.reset_index(drop=True)


'''delete redundant or wrong frames of same vehicle ID'''

data_np = data.values.astype(np.float32)

# Create empty list to organize data by Vehicle ID
sliced = []
print('---------slicing data by vehicle ID-----------')
# for i in tqdm(range(int(min(data_np[:, 0])), int(max(data_np[:, 0]))+1), ascii=True):
for i in tqdm(data['Vehicle_ID'].unique(), ascii=True):
    f_mode = find_mode(data_np, i)
    temp = data_np[data_np[:, 0] == i]
    temp = temp[temp[:, 2] == f_mode]
    sliced.append(temp)


# Create empty list for each variable,
# d0--distance from lane center, negative value means left from lane center
# road width = 12 feet
v_id, frames, dx, dy, theta, x_p, x_o, y_p, y_o, v, a, lane_num, vehicle_length, vehicle_width, Class, label, d0 = (
    [] for i in range(17))

print('------Calculating theta and add lane change label-----')
for z in tqdm(range(0, len(sliced)), ascii=True):

    # store rows for left lane change trajectories and right ones
    LK_row, LC_row, RC_row = ([] for i in range(3))

    # store x_potion ,y_potion information with format of Dataframe
    pd_x, pd_y, pd_v = pd.DataFrame(
        sliced[z][:, 3]), pd.DataFrame(sliced[z][:, 4]), pd.DataFrame(sliced[z][:, 5])
    # z contains vehicle number. Thus, sliced[1][:,2] contains x_pos
    # data for 1st vehicle, and so on.
    vehicle_ID = sliced[z][:, 0]
    frame = sliced[z][:, 1]

    ''' 
    use Exponentially Weighted Moving-Average(EWMA) to smooth position information
    EWMA function: 
          v(t)=β·y(t)+(1-β)v(t-1)
    where v(t) is the predicted value at time t
          y(t) is observation value at time t
          β = 2/(span+1)
    '''
    x_pos = pd_x.ewm(span=10).mean().values.reshape(len(frame))
    y_pos = pd_y.ewm(span=10).mean().values.reshape(len(frame))
    vel = pd_v.ewm(span=20).mean().values.reshape(len(frame))

    # store original position information for camparision
    x_pos_o = sliced[z][:, 3]
    y_pos_o = sliced[z][:, 4]
    vel_o = sliced[z][:, 5]

    lane = sliced[z][:, 6]
    v_length = sliced[z][:, 7]
    v_width = sliced[z][:, 8]
    cla = sliced[z][:, 9]
    lab = sliced[z][:, 10]

    for j in range(len(x_pos)):

        # d0 = x_position - 12*lane_ID + 6
        d0.append((x_pos[j] - 12 * lane[j]+6)/3.2808)

        # Due to noise in data, 5 time steps are used.
        # The delta_x is found by x(t) - x(t-5), same for delta_y
        # Assume orientation of 0 (parallel to lane) for first 4 timesteps of each vehicle
        if j < 5:
            dx.append(0)
            dy.append(0)
            theta.append(0)
            a.append(0)
        else:
            delta_x = x_pos[j]-x_pos[j-5]
            delta_y = y_pos[j]-y_pos[j-5]
            dx.append(delta_x)
            dy.append(delta_y)
            theta.append(math.atan2(delta_x, delta_y))  # OUTPUT: RADIANS!
            a.append(a[j-1] if j == len(x_pos) -
                     1 else 10/3.2808*(vel[j+1]-vel[j]))

            if 25 <= j < len(x_pos)-25:
                if cla[j] == 2:
                    if abs(x_pos[j-25]-x_pos[j+25]) > 2.5:
                        if lane[j] == lane[j-1]-1:
                            # left lane change label for training data extraction
                            # lane change point before and after 5 frames get labeled
                            LC_row.extend(j-19+a for a in range(20))
                        if lane[j] == lane[j-1]+1:
                            RC_row.extend(j-19+a for a in range(20))
                    if 50 <= j <= len(x_pos)-50 and lane[j-50] == lane[j+49] \
                            and abs(x_pos[j-25]-x_pos[j+25]) < 1.:
                        LK_row.extend(j-19+a for a in range(20))

    '''
    Add lane change label for car(v_class=2),
    0 for lane-keep,1 for turn-left,2 for turn-right,3 for both
    '''
    lab[list(set(LK_row))] = 0
    lab[list(set(LC_row))] = 1
    lab[list(set(RC_row))] = 2
    # lab[list(list(set(LC_row) & set(RC_row)))] = 3
    # Append back to list.
    # 1 m = 3.2808 feet
    v_id = np.append(v_id, vehicle_ID)
    frames = np.append(frames, frame)
    x_p = np.append(x_p, x_pos/3.2808)
    x_o = np.append(x_o, x_pos_o/3.2808)
    y_p = np.append(y_p, y_pos/3.2808)
    y_o = np.append(y_o, y_pos_o/3.2808)
    v = np.append(v, vel/3.2808)
    lane_num = np.append(lane_num, lane)
    vehicle_length = np.append(vehicle_length, v_length/3.2808)
    vehicle_width = np.append(vehicle_width, v_width/3.2808)
    Class = np.append(Class, cla)
    label = np.append(label, lab)

features, baselines = pd.DataFrame(), pd.DataFrame()
features = features.assign(Vehicle_ID=v_id, frame=frames, x_position=x_p, y_position=y_p, velocity=v, a=a,
                           theta=theta, lane=lane_num, L=None, R=None, v_length=vehicle_length, v_Width=vehicle_width,
                           d0=d0, v_class=Class, label=label)

# if left lane exists ,'L'=1 ; else 'L'=0.
features['L'] = 1
features.loc[features['lane'].isin([1]), 'L'] = 0
features['R'] = 1
features.loc[features['lane'].isin([7]), 'R'] = 0

features.to_csv('processed_data/us-101.3.csv')



Now we must iterate through the whole list to find d1~d6 and v1~v6.
Surrounding information for lane change decision

# create empty lists for variables,
# d -- distance from *
# v -- velocity of *
#   d4        d3
#   d2  Auto  d1
#   d6        d5
d1, d2, d3, d4, d5, d6, v1, v2, v3, v4, v5, v6 = (
    [] for i in range(12))  # iterate through whole list

# Preprocess needed data only
feature = features.copy().ix[:, [1, 2, 3, 4, 7, 14]]
processed_list = feature[feature['label'].isin([1, 2])].index.values
row = []
for j in processed_list:
    if j > 29:
        row.extend(j-29+a for a in range(30))
row = list(set(row))
row.sort()
processed_list = row

print('-----Calculating surrounding information -------')
# for i in tqdm(range(0, len(x_p)), ascii=True):
for i in tqdm(processed_list, ascii=True):
    # init values behind as zero, infront as 1000
    ve2, ve4, ve6 = (0 for f in range(3))
    di1, di2, di3, di4, di5, di6, ve1, ve3, ve5 = (1000 for n in range(9))

    # obtain relevant data
    vehicle_x = x_p[i]
    vehicle_y = y_p[i]
    frame = data_np[i, 1]
    lane = lane_num[i]
    velocity = v[i]

    # The next two lines collect data for surrounding cars at that specific time step.
    # Procedure: 1) List all vehicles in that frame 2) Leave vehicles with greater y value
    # in variable 'infront', and lower y value in variable 'behind'
    infront = feature.loc[(feature['frame'] == frame) & (feature['y_position'] > vehicle_y),
                          ['x_position', 'y_position', 'velocity', 'lane']]
    behind = feature.loc[(feature['frame'] == frame) & (feature['y_position'] < vehicle_y),
                         ['x_position', 'y_position', 'velocity', 'lane']]

    # From 'infront' and 'behind', find all vehicles in lane +1, lane, and lane-1 lanes.
    infront_same = infront.loc[(infront['lane'] == lane)]
    behind_same = behind.loc[(behind['lane'] == lane)]
    infront_higher = infront.loc[(infront['lane'] == lane+1)]
    behind_higher = behind.loc[(behind['lane'] == lane+1)]
    infront_lower = infront.loc[(infront['lane'] == lane-1)]
    behind_lower = behind.loc[(behind['lane'] == lane-1)]

    # Feature extractor. 'infront_same.empty' returns true if 'infront_same' list is empty
    # the list 'infront' returns values in increasing y_position.
    # Thus, .idxmin() is used to find the index of the closest car infront of subject vehicle.
    # .idxmax() is used to find index of closest car behind vehicle.

    if lane == 7:
        if not infront_same.empty:
            index1 = infront_same['y_position'].idxmin()
            di1 = feature['y_position'][index1]-vehicle_y
            ve1 = feature['velocity'][index1]
        if not behind_same.empty:
            index2 = behind_same['y_position'].idxmax()
            di2 = vehicle_y-feature['y_position'][index2]
            ve2 = feature['velocity'][index2]
        di5 = 0
        ve5 = velocity
        di6 = 0
        ve6 = velocity
        if not infront_lower.empty:
            index3 = infront_lower['y_position'].idxmin()
            di3 = abs(feature['y_position'][index3]-vehicle_y)
            ve3 = feature['velocity'][index3]
        if not behind_lower.empty:
            index4 = behind_lower['y_position'].idxmax()
            di4 = abs(feature['y_position'][index4]-vehicle_y)
            ve4 = feature['velocity'][index4]

    elif lane == 1:
        if not infront_same.empty:
            index1 = infront_same['y_position'].idxmin()
            di1 = feature['y_position'][index1]-vehicle_y
            ve1 = feature['velocity'][index1]
        if not behind_same.empty:
            index2 = behind_same['y_position'].idxmax()
            di2 = vehicle_y-feature['y_position'][index2]
            ve2 = feature['velocity'][index2]
        if not infront_higher.empty:
            index5 = infront_higher['y_position'].idxmin()
            di5 = abs(feature['y_position'][index5]-vehicle_y)
            ve5 = feature['velocity'][index5]
        if not behind_higher.empty:
            index6 = behind_higher['y_position'].idxmax()
            di6 = abs(feature['y_position'][index6]-vehicle_y)
            ve6 = feature['velocity'][index6]
        di3 = 0
        di4 = 0
        ve3 = velocity
        ve4 = velocity

    elif lane == 6:
        if not infront_same.empty:
            index1 = infront_same['y_position'].idxmin()
            di1 = feature['y_position'][index1]-vehicle_y
            ve1 = feature['velocity'][index1]
        if not behind_same.empty:
            index2 = behind_same['y_position'].idxmax()
            di2 = vehicle_y-feature['y_position'][index2]
            ve2 = feature['velocity'][index2]
        di5 = 0
        di6 = 0
        ve5 = velocity
        ve6 = velocity
        if not infront_lower.empty:
            index3 = infront_lower['y_position'].idxmin()
            di3 = abs(feature['y_position'][index3]-vehicle_y)
            ve3 = feature['velocity'][index3]
        if not behind_lower.empty:
            index4 = behind_lower['y_position'].idxmax()
            di4 = abs(feature['y_position'][index4]-vehicle_y)
            ve4 = feature['velocity'][index4]
    else:

        if not infront_same.empty:
            index1 = infront_same['y_position'].idxmin()
            di1 = feature['y_position'][index1]-vehicle_y
            ve1 = feature['velocity'][index1]
        if not behind_same.empty:
            index2 = behind_same['y_position'].idxmax()
            di2 = vehicle_y-feature['y_position'][index2]
            ve2 = feature['velocity'][index2]
        if not infront_higher.empty:
            index5 = infront_higher['y_position'].idxmin()
            di5 = abs(feature['y_position'][index5]-vehicle_y)
            ve5 = feature['velocity'][index5]
        if not behind_higher.empty:
            index6 = behind_higher['y_position'].idxmax()
            di6 = abs(feature['y_position'][index6]-vehicle_y)
            ve6 = feature['velocity'][index6]
        if not infront_lower.empty:
            index3 = infront_lower['y_position'].idxmin()
            di3 = abs(feature['y_position'][index3]-vehicle_y)
            ve3 = feature['velocity'][index3]
        if not behind_lower.empty:
            index4 = behind_lower['y_position'].idxmax()
            di4 = abs(feature['y_position'][index4]-vehicle_y)
            ve4 = feature['velocity'][index4]

    # append all values to list.
    features.loc[i, 'd1'] = di1
    features.loc[i, 'd2'] = di2
    features.loc[i, 'd3'] = di3
    features.loc[i, 'd4'] = di4
    features.loc[i, 'd6'] = di5
    features.loc[i, 'd6'] = di6
    features.loc[i, 'v1'] = ve1
    features.loc[i, 'v2'] = ve2
    features.loc[i, 'v3'] = ve3
    features.loc[i, 'v4'] = ve4
    features.loc[i, 'v5'] = ve5
    features.loc[i, 'v6'] = ve6


# exchange columns and related data, 'lable'--> end
cols = list(features)
cols.append(cols.pop(cols.index('label')))
features = features.loc[:, cols]
features.to_csv('processed_data/i-80.1test.csv')
