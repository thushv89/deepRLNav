import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

data_dir = 'image_probs'
prob_filename = data_dir + os.sep + 'probabilities.txt'

ra_probs = []
sd_probs = []


image_prob_list = []
with open(prob_filename, 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',')
    for row in bumpreader:
        image_prob_list.append([row[0],float(row[3]),float(row[2]),float(row[1])])

fig, axarr = plt.subplots(4,3,gridspec_kw={'height_ratios':[3,1,3,1],'hspace':0.2,'wspace':0.2})
ax1 = axarr[0][0]
ax2 = axarr[0][1]
ax3 = axarr[0][2]
ax4 = axarr[1][0]
ax5 = axarr[1][1]
ax6 = axarr[1][2]

skip_with = 2
x_axis = [0,1,2]
my_xticks = ['L','S','R']

for r_i in range(0,3,skip_with):
    for c_i in range(3):
        list_idx = (r_i//skip_with)*3+c_i
        img =  mpimg.imread(data_dir+os.sep+image_prob_list[list_idx][0]+'.png')
        axarr[r_i][c_i].imshow(img)
        axarr[r_i][c_i].axis('off')
        axarr[r_i+1][c_i].bar(x_axis,image_prob_list[list_idx][1:],align='center')
        axarr[r_i+1][c_i].set_xticks(x_axis)
        axarr[r_i+1][c_i].set_xticklabels(my_xticks)
        axarr[r_i+1][c_i].set_yticks([0,0.5,1.0])
        axarr[r_i+1][c_i].set_yticklabels([0,0.5,1.0])
        axarr[r_i+1][c_i].set_ylim([0.0,1.0])
        if c_i==0:
            axarr[r_i+1][c_i].set_ylabel('Probabilities')
        if r_i==0 and (c_i==0 or c_i==1):
            axarr[r_i][c_i].set_title('Correctly Classified')
        if r_i==0 and c_i==2:
            axarr[r_i][c_i].set_title('Misclassified')

fig.subplots_adjust(bottom=0.05,top=0.95)
#plt.tight_layout()
plt.show()