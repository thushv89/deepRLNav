import numpy as np
import matplotlib.pyplot as plt

import csv

sliding_window = True
uncertainity_per_bump = True

window_size = 5

filenames = ['bump_log_train_office_sd.log','bump_log_train_office_ra.log','bump_log_train_office_sd.log','bump_log_train_office_ra.log']
sim_ra_bump_nw = []
sim_ra_bump_w = []

with open(filenames[0], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',')
    for row in bumpreader:
        i=0
        for col in row:
            if i==1:
                sim_ra_bump_nw.append(float(col))
            elif i==2:
                sim_ra_bump_w.append(float(col))
            i+=1

sim_sd_bump_nw = []
sim_sd_bump_w = []
with open(filenames[1], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in bumpreader:
        i=0
        for col in row:
            if i==1:
                sim_sd_bump_nw.append(float(col))
            elif i==2:
                sim_sd_bump_w.append(float(col))
            i+=1

office_ra_bump_nw = []
office_ra_bump_w = []
with open(filenames[2], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in bumpreader:
        i=0
        for col in row:
            if i==1:
                office_ra_bump_nw.append(float(col))
            elif i==2:
                office_ra_bump_w.append(float(col))
            i+=1

office_sd_bump_nw = []
office_sd_bump_w = []
with open(filenames[3], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in bumpreader:
        i=0
        for col in row:
            if i==1:
                office_sd_bump_nw.append(float(col))
            elif i==2:
                office_sd_bump_w.append(float(col))
            i+=1

x_vals = np.arange(len(sim_sd_bump_nw)) * 25 + 25

def slide_window(data):
    data_slide = []
    for i,_ in enumerate(data):
        if i<window_size:
            if i==0:
                data_slide.append(data[i])
            else:
                data_slide.append(np.mean(data[:i]))
        else:
            data_slide.append(np.mean(data[i - window_size:i]))

    print(len(data), len(data_slide))

    return data_slide

if not uncertainity_per_bump:
    fig, axarr = plt.subplots(1,3)
    ax1 = axarr[0]
    ax2 = axarr[1]
    ax3 = axarr[2]
else:
    fig, axarr = plt.subplots(2,3)
    ax1 = axarr[0][0]
    ax2 = axarr[0][1]
    ax3 = axarr[0][2]
    ax4 = axarr[1][0]
    ax5 = axarr[1][1]
    ax6 = axarr[1][2]

if uncertainity_per_bump:
    sim_ra_bump_w = [sim_ra_bump_w[i]/sim_ra_bump_nw[i] for i in range(len(sim_ra_bump_w))]
    sim_sd_bump_w = [sim_sd_bump_w[i]/sim_sd_bump_nw[i] for i in range(len(sim_sd_bump_w))]

    office_ra_bump_w = [office_ra_bump_w[i]/office_ra_bump_nw[i] for i in range(len(office_ra_bump_w))]
    office_sd_bump_w = [office_sd_bump_w[i]/office_sd_bump_nw[i] for i in range(len(office_sd_bump_w))]


if not sliding_window:

    if not uncertainity_per_bump:
        ax1.plot(x_vals, sim_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax1.plot(x_vals, sim_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax1.plot(x_vals, sim_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax1.plot(x_vals, sim_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

        ax2.plot(x_vals, office_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax2.plot(x_vals, office_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

    else:
        ax1.plot(x_vals, sim_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax4.plot(x_vals, sim_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax1.plot(x_vals, sim_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax4.plot(x_vals, sim_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

        ax2.plot(x_vals, office_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax5.plot(x_vals, office_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax5.plot(x_vals, office_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

else:

    if not uncertainity_per_bump:
        sim_ra_bump_w_slide=slide_window(sim_ra_bump_w)
        sim_ra_bump_nw_slide=slide_window(sim_ra_bump_nw)
        sim_sd_bump_w_slide = slide_window(sim_sd_bump_w)
        sim_sd_bump_nw_slide = slide_window(sim_sd_bump_nw)

        office_ra_bump_w_slide=slide_window(office_ra_bump_w)
        office_ra_bump_nw_slide=slide_window(office_ra_bump_nw)
        office_sd_bump_w_slide = slide_window(office_sd_bump_w)
        office_sd_bump_nw_slide = slide_window(office_sd_bump_nw)

        ax1.plot(x_vals, sim_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax1.plot(x_vals, sim_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax1.plot(x_vals, sim_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax1.plot(x_vals, sim_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

        ax2.plot(x_vals, office_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax2.plot(x_vals, office_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

    else:
        sim_ra_bump_w_slide=slide_window(sim_ra_bump_w)
        sim_ra_bump_nw_slide=slide_window(sim_ra_bump_nw)
        sim_sd_bump_w_slide = slide_window(sim_sd_bump_w)
        sim_sd_bump_nw_slide = slide_window(sim_sd_bump_nw)

        office_ra_bump_w_slide=slide_window(office_ra_bump_w)
        office_ra_bump_nw_slide=slide_window(office_ra_bump_nw)
        office_sd_bump_w_slide = slide_window(office_sd_bump_w)
        office_sd_bump_nw_slide = slide_window(office_sd_bump_nw)

        ax1.plot(x_vals, sim_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax4.plot(x_vals, sim_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax1.plot(x_vals, sim_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax4.plot(x_vals, sim_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

        ax2.plot(x_vals, office_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax5.plot(x_vals, office_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax5.plot(x_vals, office_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

ax1.set_title('Simulation')
ax1.set_xlabel('Episode')
ax1.set_title('Simulation')
ax1.legend(fontsize=5)

ax2.set_title('Office (Real)')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Number of Bumps')
ax2.legend(fontsize=5)

if uncertainity_per_bump:
    ax4.set_title('Certainity of Mis-Classified actions')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Certainity')

    ax4.set_title('Certainity of Mis-Classified actions')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Certainity')
#plt.title('Behavior of non-weighted and Weighted bump count')
#plt.xlabel('Episode')
plt.ylabel('Number of Bumps')

legend = plt.legend(loc='center right', shadow=False, fontsize='medium')

plt.show()
