import numpy as np
import matplotlib.pyplot as plt

import csv

sliding_window = True
uncertainity_per_bump = True

window_size = 5
avg_type = 'normal'
learning_rate = 0.4

filenames = ['bump_log_train_sim_ra.log','bump_log_train_sim_sd.log','bump_log_train_office_ra.log','bump_log_train_office_sd.log','bump_log_train_outdoor_ra.log','bump_log_train_outdoor_sd.log']
sim_ra_bump_nw = []
sim_ra_bump_w = []

legend_fontsize = 8

drop_count = 2
with open(filenames[0], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',')
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue
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
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue

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
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue
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
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue
        for col in row:
            if i==1:
                office_sd_bump_nw.append(float(col))
            elif i==2:
                office_sd_bump_w.append(float(col))
            i+=1

outdoor_ra_bump_nw = []
outdoor_ra_bump_w = []
with open(filenames[4], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue
        for col in row:
            if i==1:
                outdoor_ra_bump_nw.append(float(col))
            elif i==2:
                outdoor_ra_bump_w.append(float(col))
            i+=1


outdoor_sd_bump_nw = []
outdoor_sd_bump_w = []

with open(filenames[5], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue
        for col in row:
            if i==1:
                outdoor_sd_bump_nw.append(float(col))
            elif i==2:
                outdoor_sd_bump_w.append(float(col))
            i+=1

x_vals = (25*drop_count)+ np.arange(len(office_sd_bump_nw)) * 25 + 25
out_x_vals = (25*drop_count)+ np.arange(len(outdoor_sd_bump_nw)) * 25 + 25
sim_x_vals = (25*drop_count) +  np.arange(len(sim_sd_bump_nw)) * 25 + 25

def slide_window(data,avg_type='normal'):
    data_slide = []
    for i,_ in enumerate(data):
        if i<window_size:
            if i==0:
                data_slide.append(data[i])
            else:
                if avg_type=='normal':
                    data_slide.append(np.mean(data[:i+1]))
                elif avg_type=='exp':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
        else:
            if avg_type=='normal':
                data_slide.append(np.mean(data[i - window_size+1:i+1]))
            elif avg_type=='exp':
                raise NotImplementedError
            else:
                raise NotImplementedError

    print(len(data), len(data_slide))

    return data_slide


def slide_window_v2(data):
    data_slide = []
    begin_idx = 0
    end_idx = 0
    area=-1
    for i,_ in enumerate(data):
        if i<=window_size//2:
            area = 0
            begin_idx = 0
            end_idx = i+i #if current index does not fit in a complet window we choose the window to be the max possible size
        elif i>=len(data)-(window_size//2):
            area = 2
            begin_idx = i-(len(data)-1 - i)
            end_idx = len(data)-1
        else:
            area = 3
            begin_idx  = i - (window_size//2)
            end_idx = i+(window_size//2)
        print('For data point at %i range: [%i,%i] (Area %i)'%(i,begin_idx,end_idx,area))
        if avg_type=='normal':
            data_slide.append(np.mean(data[begin_idx:end_idx+1]))
        elif avg_type=='exp':
            data_slide.append(get_exp_average(data[begin_idx:end_idx+1]))
    print(len(data), len(data_slide))

    return data_slide

def get_exp_average(data):
    mid_val = data[len(data)//2]
    avg = mid_val
    for i in range((len(data)//2)):
        avg = learning_rate*avg + (1-learning_rate)*(data[(len(data)//2)-i]+data[(len(data)//2)+i])/2

    print('Mid val is %.2f and Exp Average is %.2f'%(mid_val,avg))
    return avg

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

    outdoor_ra_bump_w = [outdoor_ra_bump_w[i]/outdoor_ra_bump_nw[i] for i in range(len(outdoor_ra_bump_w))]
    outdoor_sd_bump_w = [outdoor_sd_bump_w[i]/outdoor_sd_bump_nw[i] for i in range(len(outdoor_sd_bump_w))]

if not sliding_window:

    if not uncertainity_per_bump:
        ax1.plot(sim_x_vals, sim_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax1.plot(sim_x_vals, sim_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax1.plot(sim_x_vals, sim_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax1.plot(sim_x_vals, sim_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

        ax2.plot(x_vals, office_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax2.plot(x_vals, office_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax3.plot(out_x_vals, outdoor_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax3.plot(out_x_vals, outdoor_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

    else:
        ax1.plot(sim_x_vals, sim_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax4.plot(sim_x_vals, sim_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax1.plot(sim_x_vals, sim_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax4.plot(sim_x_vals, sim_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

        ax2.plot(x_vals, office_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax5.plot(x_vals, office_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax5.plot(x_vals, office_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax6.plot(out_x_vals, outdoor_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax6.plot(out_x_vals, outdoor_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')

else:

    if not uncertainity_per_bump:
        sim_ra_bump_w_slide=slide_window_v2(sim_ra_bump_w)
        sim_ra_bump_nw_slide=slide_window_v2(sim_ra_bump_nw)
        sim_sd_bump_w_slide = slide_window_v2(sim_sd_bump_w)
        sim_sd_bump_nw_slide = slide_window_v2(sim_sd_bump_nw)

        office_ra_bump_w_slide=slide_window_v2(office_ra_bump_w)
        office_ra_bump_nw_slide=slide_window_v2(office_ra_bump_nw)
        office_sd_bump_w_slide = slide_window_v2(office_sd_bump_w)
        office_sd_bump_nw_slide = slide_window_v2(office_sd_bump_nw)

        outdoor_ra_bump_w_slide=slide_window_v2(outdoor_ra_bump_w)
        outdoor_ra_bump_nw_slide=slide_window_v2(outdoor_ra_bump_nw)
        outdoor_sd_bump_w_slide = slide_window_v2(outdoor_sd_bump_w)
        outdoor_sd_bump_nw_slide = slide_window_v2(outdoor_sd_bump_nw)

        ax1.plot(sim_x_vals, sim_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax1.plot(sim_x_vals, sim_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax1.plot(sim_x_vals, sim_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax1.plot(sim_x_vals, sim_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

        ax2.plot(x_vals, office_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax2.plot(x_vals, office_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax3.plot(out_x_vals, outdoor_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax3.plot(out_x_vals, outdoor_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

    else:
        sim_ra_bump_w_slide=slide_window_v2(sim_ra_bump_w)
        sim_ra_bump_nw_slide=slide_window_v2(sim_ra_bump_nw)
        sim_sd_bump_w_slide = slide_window_v2(sim_sd_bump_w)
        sim_sd_bump_nw_slide = slide_window_v2(sim_sd_bump_nw)

        office_ra_bump_w_slide=slide_window_v2(office_ra_bump_w)
        office_ra_bump_nw_slide=slide_window_v2(office_ra_bump_nw)
        office_sd_bump_w_slide = slide_window_v2(office_sd_bump_w)
        office_sd_bump_nw_slide = slide_window_v2(office_sd_bump_nw)

        outdoor_ra_bump_w_slide=slide_window_v2(outdoor_ra_bump_w)
        outdoor_ra_bump_nw_slide=slide_window_v2(outdoor_ra_bump_nw)
        outdoor_sd_bump_w_slide = slide_window_v2(outdoor_sd_bump_w)
        outdoor_sd_bump_nw_slide = slide_window_v2(outdoor_sd_bump_nw)

        ax1.plot(sim_x_vals, sim_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax4.plot(sim_x_vals, sim_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax1.plot(sim_x_vals, sim_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax4.plot(sim_x_vals, sim_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

        ax2.plot(x_vals, office_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax5.plot(x_vals, office_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax5.plot(x_vals, office_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax6.plot(out_x_vals, outdoor_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax6.plot(out_x_vals, outdoor_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')

ax1.set_title('Simulation')
ax1.set_xlabel('Episode')
ax1.set_title('Simulation')
ax1.legend(fontsize=legend_fontsize)

ax2.set_title('Office (Real)')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Number of Bumps')
ax2.legend(fontsize=legend_fontsize)

ax3.set_title('Outdoor (Real)')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Number of Bumps')
ax3.legend(fontsize=legend_fontsize)

if uncertainity_per_bump:
    ax4.set_title('Certainity of Wrong Actions')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Certainity')
    ax4.legend(fontsize=legend_fontsize)

    ax5.set_title('Certainity of Wrong Actions')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Certainity')
    ax5.legend(fontsize=legend_fontsize)

    ax6.set_title('Certainity of Wrong Actions')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Certainity')
    ax6.legend(fontsize=legend_fontsize)

#plt.title('Behavior of non-weighted and Weighted bump count')
#plt.xlabel('Episode')
plt.ylabel('Number of Bumps')

#legend = plt.legend(loc='center right', shadow=False, fontsize='medium')

fig.subplots_adjust(wspace=.2,hspace=0.4,bottom=0.2)
#plt.tight_layout()
plt.show()
