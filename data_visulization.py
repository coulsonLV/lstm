import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import time



filepath = 'processed_data/data4visual.csv'
data = pd.read_csv(filepath)
data_cut = data[['Vehicle_ID', 'frame', 'x_position',
                 'y_position', 'lane', 'v_length', 'v_Width', 'theta']]
sorted_frame = data_cut.sort_values(by=['frame'])
sorted_np = sorted_frame.values
sorted_np = sorted_np[0:50000, :]

# init array of sliced values, by frame number
sliced = []
# slice data by frame number
for i in range(int(min(sorted_np[:, 1])), int(max(sorted_np[:, 1]))):
    sliced.append(sorted_np[sorted_np[:, 1] == i])


fig = plt.figure(figsize=(18, 3))
ax = fig.add_subplot(1, 1, 1)


def animate(i):

    names = sliced[i][:, 0]
    x = sliced[i][:, 2]*3.2808
    y = sliced[i][:, 3]*3.2808
    lane = sliced[i][:, 4]
    vehicle_length = sliced[i][:, 5]*3.2808
    vehicle_width = sliced[i][:, 6]*3.2808

    ax.clear()
    plt.axhline(y=12, color='white', linestyle='--')
    plt.axhline(y=24, color='white', linestyle='--')
    plt.axhline(y=36, color='white', linestyle='--')
    plt.axhline(y=48, color='white', linestyle='--')
    plt.axhline(y=60, color='white', linestyle='--')

    # set autoscale off, set x,y axis
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    ax.set_xlim([200, 1000])
    ax.set_ylim([0, 80])
    ax.set_facecolor('gray')

    patches = []
    lane_color = ["white", "red", "orange", "yellow", "green", "blue"]

    for vid, x_, y_, lane, vlength, vwidth in zip(names, x, y, lane, vehicle_length, vehicle_width):
        # print(x_cent, y_cent)
        vlen = vlength*0.75
        vwid = vwidth*0.75
        patches.append(ax.add_patch(plt.Rectangle((y_-vlen/2, x_-vwid/2), vlen, vwid,
                                                  fill=True, angle=0, linewidth=2, color=lane_color[int(lane)])))
        # plt.annotate(xy=(y_-vlen/2, x_-vwid/2), s=str(vid), color='r')
    # time.sleep(0.1)
    return patches


ani = animation.FuncAnimation(
    fig, animate, frames=range(2, 10000), interval=10, blit=True)
# ani.save('video.mp4')

plt.show()
