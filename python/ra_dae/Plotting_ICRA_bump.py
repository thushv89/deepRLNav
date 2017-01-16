import numpy as np
import matplotlib.pyplot as plt

import csv

sliding_window = True
uncertainity_per_bump = True

window_size = 5
avg_type = 'normal'
learning_rate = 0.4

filenames = ['bump_log_train_sim_ra.log','bump_log_train_sim_sd.log','bump_log_train_sim_lr.log','bump_log_train_office_ra.log','bump_log_train_office_sd.log','bump_log_train_office_lr.log','bump_log_train_outdoor_ra.log','bump_log_train_outdoor_sd.log','bump_log_train_outdoor_lr.log']
sim_ra_bump_nw = []
sim_ra_bump_w = []

legend_fontsize = 12
axis_fontsize=14
title_fontsize=18

only_upper_row = True

if only_upper_row:
    y_lims = [(4,16),(8,14),(6,14)]
    x_lims = [(0,500),(0,400),(0,400)]
else:
    if uncertainity_per_bump:
        y_lims = [(4,16),(8,16),(4,14),(0.1,0.9),(0.3,0.65),(0.3,0.8)]
        x_lims = [(0,500),(0,400),(0,400),(0,500),(0,400),(0,400)]

drop_count = 3
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


sim_lr_bump_nw = []
sim_lr_bump_w = []
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
                sim_lr_bump_nw.append(float(col))
            elif i==2:
                sim_lr_bump_w.append(float(col))
            i+=1


office_ra_bump_nw = []
office_ra_bump_w = []
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
                office_ra_bump_nw.append(float(col))
            elif i==2:
                office_ra_bump_w.append(float(col))
            i+=1


office_sd_bump_nw = []
office_sd_bump_w = []

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
                office_sd_bump_nw.append(float(col))
            elif i==2:
                office_sd_bump_w.append(float(col))
            i+=1

office_lr_bump_nw = []
office_lr_bump_w = []

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
                office_lr_bump_nw.append(float(col))
            elif i==2:
                office_lr_bump_w.append(float(col))
            i+=1

outdoor_ra_bump_nw = []
outdoor_ra_bump_w = []
with open(filenames[6], 'rt') as csvfile:
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

with open(filenames[7], 'rt') as csvfile:
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


outdoor_lr_bump_nw = []
outdoor_lr_bump_w = []

with open(filenames[8], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    j=0
    for row in bumpreader:
        i=0
        if j<drop_count:
            j+=1
            continue
        for col in row:
            if i==1:
                outdoor_lr_bump_nw.append(float(col))
            elif i==2:
                outdoor_lr_bump_w.append(float(col))
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
            #end_idx = i+i #if current index does not fit in a complet window we choose the window to be the max possible size
            end_idx = i+2
        elif i>=len(data)-(window_size//2):
            area = 2
            #begin_idx = i-(len(data)-1 - i)
            begin_idx = i-2
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

def get_exp_average_stdev_single(data,lr):
    avg = data[-1]
    for i in reversed(range((len(data))//2,len(data))):
        avg = lr*avg + (1-lr)*data[i]

    return avg, np.std(data[len(data)//2:])

if not uncertainity_per_bump or only_upper_row:
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
    sim_lr_bump_w = [sim_lr_bump_w[i]/sim_lr_bump_nw[i] for i in range(len(sim_lr_bump_w))]

    office_ra_bump_w = [office_ra_bump_w[i]/office_ra_bump_nw[i] for i in range(len(office_ra_bump_w))]
    office_sd_bump_w = [office_sd_bump_w[i]/office_sd_bump_nw[i] for i in range(len(office_sd_bump_w))]
    office_lr_bump_w = [office_lr_bump_w[i]/office_lr_bump_nw[i] for i in range(len(office_lr_bump_w))]

    outdoor_ra_bump_w = [outdoor_ra_bump_w[i]/outdoor_ra_bump_nw[i] for i in range(len(outdoor_ra_bump_w))]
    outdoor_sd_bump_w = [outdoor_sd_bump_w[i]/outdoor_sd_bump_nw[i] for i in range(len(outdoor_sd_bump_w))]
    outdoor_lr_bump_w = [outdoor_lr_bump_w[i]/outdoor_lr_bump_nw[i] for i in range(len(outdoor_lr_bump_w))]

    lr = 0.75
    sim_ra_avg,sim_ra_std = get_exp_average_stdev_single(sim_ra_bump_w,lr)
    sim_sd_avg,sim_sd_std = get_exp_average_stdev_single(sim_sd_bump_w,lr)
    sim_lr_avg,sim_lr_std = get_exp_average_stdev_single(sim_lr_bump_w,lr)

    office_ra_avg,office_ra_std = get_exp_average_stdev_single(office_ra_bump_w,lr)
    office_sd_avg,office_sd_std = get_exp_average_stdev_single(office_sd_bump_w,lr)
    office_lr_avg,office_lr_std = get_exp_average_stdev_single(office_lr_bump_w,lr)

    outdoor_ra_avg,outdoor_ra_std = get_exp_average_stdev_single(outdoor_ra_bump_w,lr)
    outdoor_sd_avg,outdoor_sd_std = get_exp_average_stdev_single(outdoor_sd_bump_w,lr)
    outdoor_lr_avg,outdoor_lr_std = get_exp_average_stdev_single(outdoor_lr_bump_w,lr)

    print('Single average stdev')
    print('RA-DAE')
    print('%.2f,%.2f'%(sim_ra_avg,sim_ra_std))
    print('%.2f,%.2f'%(office_ra_avg,office_ra_std))
    print('%.2f,%.2f'%(outdoor_ra_avg,outdoor_ra_std))
    print('SDAE')
    print('%.2f,%.2f'%(sim_sd_avg,sim_sd_std))
    print('%.2f,%.2f'%(office_sd_avg,office_sd_std))
    print('%.2f,%.2f'%(outdoor_sd_avg,outdoor_sd_std))

if not sliding_window:

    if not uncertainity_per_bump or only_upper_row:
        ax1.plot(sim_x_vals, sim_ra_bump_nw, 'r', label='$L_{NW}$ (RA-DAE)')
        ax1.plot(sim_x_vals, sim_ra_bump_w, 'r--', label='$L_{W}$ (RA-DAE)')
        ax1.plot(sim_x_vals, sim_sd_bump_nw, 'y', label='$L_{NW}$ (SDAE)')
        ax1.plot(sim_x_vals, sim_sd_bump_w, 'y--', label='$L_{W}$ (SDAE)')
        ax1.plot(sim_x_vals, sim_lr_bump_nw, 'b', label='$L_{NW}$ (LR)')
        ax1.plot(sim_x_vals, sim_lr_bump_w, 'b--', label='Weighted bumps (LR)')

        ax2.plot(x_vals, office_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax2.plot(x_vals, office_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')
        ax2.plot(x_vals, office_lr_bump_nw, 'b', label='Non-weighted bumps (LR)')
        ax2.plot(x_vals, office_lr_bump_w, 'b--', label='Weighted bumps (LR)')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax3.plot(out_x_vals, outdoor_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax3.plot(out_x_vals, outdoor_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')
        ax3.plot(out_x_vals, outdoor_lr_bump_nw, 'b', label='Non-weighted bumps (LR)')
        ax3.plot(out_x_vals, outdoor_lr_bump_w, 'b--', label='Weighted bumps (LR)')

    else:
        ax1.plot(sim_x_vals, sim_ra_bump_nw, 'r', label='$L_{NW}$ (RA-DAE)')
        ax4.plot(sim_x_vals, sim_ra_bump_w, 'r--', label='$L_{W}$ (RA-DAE)')
        ax1.plot(sim_x_vals, sim_sd_bump_nw, 'y', label='$L_{NW}$ (SDAE)')
        ax4.plot(sim_x_vals, sim_sd_bump_w, 'y--', label='$L_{W}$ (SDAE)')
        ax1.plot(sim_x_vals, sim_lr_bump_nw, 'b', label='$L_{NW}$ (LR)')
        ax4.plot(sim_x_vals, sim_lr_bump_w, 'b--', label='$L_{W}$ (LR)')

        ax2.plot(x_vals, office_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax5.plot(x_vals, office_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax2.plot(x_vals, office_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax5.plot(x_vals, office_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')
        ax2.plot(x_vals, office_lr_bump_nw, 'b', label='Non-weighted bumps (LR)')
        ax5.plot(x_vals, office_lr_bump_w, 'b--', label='Weighted bumps (LR)')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw, 'r', label='Non-weighted bumps (RA-DAE)')
        ax6.plot(out_x_vals, outdoor_ra_bump_w, 'r--', label='Weighted bumps (RA-DAE)')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw, 'y', label='Non-weighted bumps (SDAE)')
        ax6.plot(out_x_vals, outdoor_sd_bump_w, 'y--', label='Weighted bumps (SDAE)')
        ax3.plot(out_x_vals, outdoor_lr_bump_nw, 'b', label='Non-weighted bumps (LR)')
        ax6.plot(out_x_vals, outdoor_lr_bump_w, 'b--', label='Weighted bumps (LR)')

else:

    if not uncertainity_per_bump or only_upper_row:
        sim_ra_bump_w_slide=slide_window_v2(sim_ra_bump_w)
        sim_ra_bump_nw_slide=slide_window_v2(sim_ra_bump_nw)
        sim_sd_bump_w_slide = slide_window_v2(sim_sd_bump_w)
        sim_sd_bump_nw_slide = slide_window_v2(sim_sd_bump_nw)
        sim_lr_bump_w_slide = slide_window_v2(sim_lr_bump_w)
        sim_lr_bump_nw_slide = slide_window_v2(sim_lr_bump_nw)

        office_ra_bump_w_slide=slide_window_v2(office_ra_bump_w)
        office_ra_bump_nw_slide=slide_window_v2(office_ra_bump_nw)
        office_sd_bump_w_slide = slide_window_v2(office_sd_bump_w)
        office_sd_bump_nw_slide = slide_window_v2(office_sd_bump_nw)
        office_lr_bump_w_slide = slide_window_v2(office_lr_bump_w)
        office_lr_bump_nw_slide = slide_window_v2(office_lr_bump_nw)

        outdoor_ra_bump_w_slide=slide_window_v2(outdoor_ra_bump_w)
        outdoor_ra_bump_nw_slide=slide_window_v2(outdoor_ra_bump_nw)
        outdoor_sd_bump_w_slide = slide_window_v2(outdoor_sd_bump_w)
        outdoor_sd_bump_nw_slide = slide_window_v2(outdoor_sd_bump_nw)
        outdoor_lr_bump_w_slide = slide_window_v2(outdoor_lr_bump_w)
        outdoor_lr_bump_nw_slide = slide_window_v2(outdoor_lr_bump_nw)

        ax1.plot(sim_x_vals, sim_ra_bump_nw_slide, 'r', label='$L_{NW}$ (RA-DAE)',marker='o')
        ax1.plot(sim_x_vals, sim_sd_bump_nw_slide, 'b', label='$L_{NW}$ (SDAE)',marker='x')
        ax1.plot(sim_x_vals, sim_lr_bump_nw_slide, 'y', label='$L_{NW}$ (LR)',marker='s')

        if not only_upper_row:
            ax1.plot(sim_x_vals, sim_ra_bump_w_slide, 'r--', label='$L_{W}$ (RA-DAE)',marker='o')
            ax1.plot(sim_x_vals, sim_sd_bump_w_slide, 'b--', label='$L_{W}$ (SDAE)',marker='x')
            ax1.plot(sim_x_vals, sim_lr_bump_w_slide, 'y', label='$L_{W}$ (LR)',marker='s')

        ax2.plot(x_vals, office_ra_bump_nw_slide, 'r', label='$L_{NW}$ (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_sd_bump_nw_slide, 'b', label='$L_{NW}$ (SDAE)',marker='x')
        ax2.plot(x_vals, office_lr_bump_nw_slide, 'y', label='$L_{NW}$ (LR)',marker='s')
        if not only_upper_row:
            ax2.plot(x_vals, office_ra_bump_w_slide, 'r--', label='$L_{W}$ (RA-DAE)',marker='o')
            ax2.plot(x_vals, office_sd_bump_w_slide, 'b--', label='$L_{W}$ (SDAE)',marker='x')
            ax2.plot(x_vals, office_lr_bump_w_slide, 'y--', label='$L_{W}$ (LR)',marker='s')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw_slide, 'r', label='$L_{NW}$ (RA-DAE)',marker='o')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw_slide, 'b', label='$L_{NW}$ (SDAE)',marker='x')
        ax3.plot(out_x_vals, outdoor_lr_bump_nw_slide, 'y', label='$L_{NW}$ (LR)',marker='s')
        if not only_upper_row:
            ax3.plot(out_x_vals, outdoor_ra_bump_w_slide, 'r--', label='$L_{W}$ (RA-DAE)',marker='o')
            ax3.plot(out_x_vals, outdoor_sd_bump_w_slide, 'b--', label='$L_{W}$ (SDAE)',marker='x')
            ax3.plot(out_x_vals, outdoor_lr_bump_w_slide, 'y--', label='$L_{W}$ (LR)',marker='s')

    else:
        sim_ra_bump_w_slide=slide_window_v2(sim_ra_bump_w)
        sim_ra_bump_nw_slide=slide_window_v2(sim_ra_bump_nw)
        sim_sd_bump_w_slide = slide_window_v2(sim_sd_bump_w)
        sim_sd_bump_nw_slide = slide_window_v2(sim_sd_bump_nw)
        sim_lr_bump_w_slide = slide_window_v2(sim_lr_bump_w)
        sim_lr_bump_nw_slide = slide_window_v2(sim_lr_bump_nw)

        office_ra_bump_w_slide=slide_window_v2(office_ra_bump_w)
        office_ra_bump_nw_slide=slide_window_v2(office_ra_bump_nw)
        office_sd_bump_w_slide = slide_window_v2(office_sd_bump_w)
        office_sd_bump_nw_slide = slide_window_v2(office_sd_bump_nw)
        office_lr_bump_w_slide = slide_window_v2(office_lr_bump_w)
        office_lr_bump_nw_slide = slide_window_v2(office_lr_bump_nw)

        outdoor_ra_bump_w_slide=slide_window_v2(outdoor_ra_bump_w)
        outdoor_ra_bump_nw_slide=slide_window_v2(outdoor_ra_bump_nw)
        outdoor_sd_bump_w_slide = slide_window_v2(outdoor_sd_bump_w)
        outdoor_sd_bump_nw_slide = slide_window_v2(outdoor_sd_bump_nw)
        outdoor_lr_bump_w_slide = slide_window_v2(outdoor_lr_bump_w)
        outdoor_lr_bump_nw_slide = slide_window_v2(outdoor_lr_bump_nw)

        ax1.plot(sim_x_vals, sim_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax4.plot(sim_x_vals, sim_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax1.plot(sim_x_vals, sim_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax4.plot(sim_x_vals, sim_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')
        ax1.plot(sim_x_vals, sim_lr_bump_nw_slide, 'y', label='Non-weighted bumps (LR)',marker='s')
        ax4.plot(sim_x_vals, sim_lr_bump_w_slide, 'y--', label='Weighted bumps (LR)',marker='s')

        ax2.plot(x_vals, office_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax5.plot(x_vals, office_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax2.plot(x_vals, office_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax5.plot(x_vals, office_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')
        ax2.plot(x_vals, office_lr_bump_nw_slide, 'y', label='Non-weighted bumps (LR)',marker='s')
        ax5.plot(x_vals, office_lr_bump_w_slide, 'y--', label='Weighted bumps (LR)',marker='s')

        ax3.plot(out_x_vals, outdoor_ra_bump_nw_slide, 'r', label='Non-weighted bumps (RA-DAE)',marker='o')
        ax6.plot(out_x_vals, outdoor_ra_bump_w_slide, 'r--', label='Weighted bumps (RA-DAE)',marker='o')
        ax3.plot(out_x_vals, outdoor_sd_bump_nw_slide, 'b', label='Non-weighted bumps (SDAE)',marker='x')
        ax6.plot(out_x_vals, outdoor_sd_bump_w_slide, 'b--', label='Weighted bumps (SDAE)',marker='x')
        ax3.plot(out_x_vals, outdoor_lr_bump_nw_slide, 'y', label='Non-weighted bumps (LR)',marker='s')
        ax6.plot(out_x_vals, outdoor_lr_bump_w_slide, 'y--', label='Weighted bumps (LR)',marker='s')

ax1.set_title('Simulation',fontsize=title_fontsize)
ax1.set_xlabel('Episode',fontsize=axis_fontsize)
ax1.set_ylabel('Number of Bumps',fontsize=axis_fontsize)
ax1.legend(fontsize=legend_fontsize)

ax2.set_title('Office (Real)',fontsize=title_fontsize)
ax2.set_xlabel('Episode',fontsize=axis_fontsize)
ax2.set_ylabel('Number of Bumps',fontsize=axis_fontsize)
ax2.legend(fontsize=legend_fontsize)

ax3.set_title('Outdoor (Real)',fontsize=title_fontsize)
ax3.set_xlabel('Episode',fontsize=axis_fontsize)
ax3.set_ylabel('Number of Bumps',fontsize=axis_fontsize)
ax3.legend(fontsize=legend_fontsize)

ax1.set_ylim(y_lims[0])
ax2.set_ylim(y_lims[1])
ax3.set_ylim(y_lims[2])
ax1.set_xlim(x_lims[0])
ax2.set_xlim(x_lims[1])
ax3.set_xlim(x_lims[2])

if uncertainity_per_bump and (not only_upper_row):

    ax4.set_title('Certainty of Wrong Actions')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Certainity')
    ax4.legend(fontsize=legend_fontsize)

    ax5.set_title('Certainty of Wrong Actions')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Certainity')
    ax5.legend(fontsize=legend_fontsize)

    ax6.set_title('Certainty of Wrong Actions')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Certainity')
    ax6.legend(fontsize=legend_fontsize)

    ax4.set_ylim(y_lims[3])
    ax5.set_ylim(y_lims[4])
    ax6.set_ylim(y_lims[5])

#plt.title('Behavior of non-weighted and Weighted bump count')
#plt.xlabel('Episode')
plt.ylabel('Number of Bumps')

#legend = plt.legend(loc='center right', shadow=False, fontsize='medium')

fig.subplots_adjust(wspace=.2,hspace=0.4,bottom=0.2)
#plt.tight_layout()
plt.show()
