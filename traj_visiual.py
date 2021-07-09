import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

traj = pd.read_csv('processed_data/data4comparison.csv')

color = ['y', 'r', 'b', 'k', 'g']
line = ['--',  '-.']

ax = traj.loc[traj['Vehicle_ID'] == int(min(traj['Vehicle_ID']))].plot(x='y_p', y='x_p',
                                                                       figsize=(15, 8), style='k',)
# for i in tqdm(range(int(min(traj['Vehicle_ID']))+1, int(max(traj['Vehicle_ID']))+1), ascii=True):
for i in tqdm(range(0, 100), ascii=True):
    if i in traj['Vehicle_ID'].values:
        style = np.random.choice(color)+np.random.choice(line)
        traj_ = traj.loc[traj['Vehicle_ID'] == i][['x_p', 'y_p']]
        traj_.plot(x='y_p', y='x_p', style=style,
                   figsize=(15, 8),  ax=ax)
        # traj.loc[traj['Vehicle_ID'] == i][['x_position',
        #                                    'y_position']] = traj_.ewm(span=10).mean()

        # traj_.plot(x='y_position', y='x_position',
        #            figsize=(15, 8), style='r-', label='EMA', ax=ax)

# more information about dataframe.plot in
# https://blog.csdn.net/u013084616/article/details/79064408



ax.set_facecolor('gray')
ax.set_xlim([60, 460])
ax.set_ylim([0, 20])

plt.axhline(y=12/3.28, color='white', linestyle='--')
plt.axhline(y=24/3.28, color='white', linestyle='--')
plt.axhline(y=36/3.28, color='white', linestyle='--')
plt.axhline(y=48/3.28, color='white', linestyle='--')
plt.axhline(y=60/3.28, color='white', linestyle='--')

# plt.legend()
plt.show()
