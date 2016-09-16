import numpy as np
import matplotlib.pyplot as plt

import csv

sliding_window = True
window_size = 5

ra_bump_nw = []
ra_bump_w = []
with open('bump_log_train_ra.log', 'rb') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in bumpreader:
        i=0
        for col in row:
            if i==1:
                ra_bump_nw.append(int(col))
            elif i==2:
                ra_bump_w.append(float(col))
            i+=1

sd_bump_nw = []
sd_bump_w = []
with open('bump_log_train_sd.log', 'rb') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in bumpreader:
        i=0
        for col in row:
            if i==1:
                sd_bump_nw.append(int(col))
            elif i==2:
                sd_bump_w.append(float(col))
            i+=1

x_vals = np.arange(len(sd_bump_nw))*25

if not sliding_window:
    plt.plot(x_vals, ra_bump_nw, 'r',label='Non-weighted bumps (RA-DAE)')
    plt.plot(x_vals, ra_bump_w, 'r--',label='Weighted bumps (RA-DAE)')

else:
    ra_bump_w_slide=[]
    ra_bump_nw_slide=[]
    sd_bump_w_slide = []
    sd_bump_nw_slide = []
    for i,_ in enumerate(ra_bump_w):
        if i<window_size:
            if i==0:
                ra_bump_nw_slide.append(ra_bump_nw[i])
                ra_bump_w_slide.append(ra_bump_w[i])
                sd_bump_nw_slide.append(sd_bump_nw[i])
                sd_bump_w_slide.append(sd_bump_w[i])
            else:
                ra_bump_nw_slide.append(np.mean(ra_bump_nw[:i]))
                ra_bump_w_slide.append(np.mean(ra_bump_w[:i]))
                sd_bump_nw_slide.append(np.mean(sd_bump_nw[:i]))
                sd_bump_w_slide.append(np.mean(sd_bump_w[:i]))
        else:
            ra_bump_nw_slide.append(np.mean(ra_bump_nw[i-window_size:i]))
            ra_bump_w_slide.append(np.mean(ra_bump_w[i-window_size:i]))
            sd_bump_nw_slide.append(np.mean(sd_bump_nw[i-window_size:i]))
            sd_bump_w_slide.append(np.mean(sd_bump_w[i-window_size:i]))

        print(len(ra_bump_nw),len(ra_bump_nw_slide))
        print(len(ra_bump_w), len(ra_bump_w_slide))

    plt.plot(x_vals, ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)')
    plt.plot(x_vals, ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)')
    plt.plot(x_vals, sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)')
    plt.plot(x_vals, sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)')

plt.title('Behavior of non-weighted and Weighted bump count')
plt.xlabel('Episode')
plt.ylabel('Number of Bumps')

legend = plt.legend(loc='center right', shadow=False, fontsize='medium')

plt.show()
