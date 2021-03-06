import numpy as np
import matplotlib.pyplot as plt

N = 1
raMeans = (0.497, 0.0024*1000, 1.064, 0.008*1000)
raStd = (0.476, 0.0005*1000, 1.965, 0.002*1000)

sdMeans = (0.884, 0.005*1000, 2.443, 0.022*1000)
sdStd = (0.278, 0.001*1000, 2.15, 0.003*1000)

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, axarr = plt.subplots(nrows=1,ncols=4)

ax1,ax2,ax3,ax4 = axarr[0],axarr[1],axarr[2],axarr[3]
rects1 = ax1.bar(ind, (raMeans[0]), width, color='r', yerr=(raStd[0]))
rects2 = ax1.bar(ind + width, (sdMeans[0]), width, color='y', yerr=(sdStd[0]))

# add some text for labels, title and axes ticks
ax1.set_ylabel('Time (s)',fontsize=20)
ax1.yaxis.set_label_coords(-0.16,0.5)
ax1.set_title('Training\n(Simulation)',fontsize=20) #,y=-0.12
ax1.set_xlabel('(a)',fontsize=20)
ax1.set_xticks([])
ax1.set_ylim([0,1.4])
#ax1.set_xticklabels(('Training time (s)'))
ax1.legend(('RA-DAE', 'SDAE'),loc=2)
ax1.tick_params(axis='both', which='major', labelsize=16)

rects3 = ax2.bar(ind, (raMeans[1]), width, color='r', yerr=(raStd[1]))
rects4 = ax2.bar(ind + width, (sdMeans[1]), width, color='y', yerr=(sdStd[1]))

# add some text for labels, title and axes ticks
ax2.set_ylabel('Time (ms)',fontsize=20)
ax2.yaxis.set_label_coords(-0.1,0.5)
ax2.set_title('Prediction\n(Simulation)',fontsize=20)
ax2.set_xlabel('(b)',fontsize=20)
ax2.set_xticks([])
ax2.set_ylim([0,7])
#ax2.set_xticklabels(('Prediction time (s)'))
ax2.legend(('RA-DAE', 'SDAE'),loc=2)
ax2.tick_params(axis='both', which='major', labelsize=16)

rects5 = ax3.bar(ind, (raMeans[2]), width, color='r', yerr=(raStd[2]))
rects6 = ax3.bar(ind + width, (sdMeans[2]), width, color='y', yerr=(sdStd[2]))
# add some text for labels, title and axes ticks
ax3.set_ylabel('Time (s)',fontsize=20)
ax3.yaxis.set_label_coords(-0.1,0.5)
ax3.set_title('Training\n(Real)',fontsize=20)
ax3.set_xlabel('(c)',fontsize=20)
ax3.set_ylim([-1,6])
ax3.set_xticks([])
#ax3.set_xticklabels(('Prediction time (s)'))
ax3.legend(('RA-DAE', 'SDAE'),loc=2)
ax3.tick_params(axis='both', which='major', labelsize=16)

rects7 = ax4.bar(ind, (raMeans[3]), width, color='r', yerr=(raStd[3]))
rects8 = ax4.bar(ind + width, (sdMeans[3]), width, color='y', yerr=(sdStd[3]))
# add some text for labels, title and axes ticks
ax4.set_ylabel('Time (ms)',fontsize=20)
ax4.yaxis.set_label_coords(-0.16,0.5)
ax4.set_title('Prediction\n(Real)',fontsize=20)
ax4.set_xlabel('(d)',fontsize=20)
ax4.set_ylim([0,35])
ax4.set_xticks([])
#ax4.set_xticklabels(('Prediction time (s)'))
#ax4.legend( ('RA-DAE', 'SDAE'),loc=2,bbox_to_anchor=(1.05, 1))
ax4.legend( ('RA-DAE', 'SDAE'),loc=2)
ax4.tick_params(axis='both', which='major', labelsize=16)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#plt.subplot(1,4,3)
fig.suptitle('Average Training and Prediction Time per Episode (RA-DAE vs SDAE)',fontsize=22,y=0.98)
#autolabel(rects1)
#autolabel(rects2)
plt.tight_layout()
plt.show()
