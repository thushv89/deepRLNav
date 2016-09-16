import numpy as np
import matplotlib.pyplot as plt

N = 1
raMeans = (0.497, 0.0024)
raStd = (0.476, 0.0005)

sdMeans = (0.884, 0.005)
sdStd = (0.278, 0.001)

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, (ax1,ax2) = plt.subplots(1,2)
rects1 = ax1.bar(ind, (raMeans[0]), width, color='r', yerr=(raStd[0]))


rects2 = ax1.bar(ind + width, (sdMeans[0]), width, color='y', yerr=(sdStd[0]))

# add some text for labels, title and axes ticks
ax1.set_ylabel('Time (s)')
ax1.set_title('Average training time per episode')
ax1.set_xticks([])
#ax1.set_xticklabels(('Training time (s)'))

ax1.legend((rects1[0], rects2[0]), ('RA-DAE', 'SDAE'),loc=4)


rects3 = ax2.bar(ind, (raMeans[1]), width, color='r', yerr=(raStd[1]))
rects4 = ax2.bar(ind + width, (sdMeans[1]), width, color='y', yerr=(sdStd[1]))

# add some text for labels, title and axes ticks
ax2.set_ylabel('Time (s)')
ax2.set_title('Average prediction time per episode')
ax2.set_xticks([])
#ax2.set_xticklabels(('Prediction time (s)'))

ax2.legend((rects1[0], rects2[0]), ('RA-DAE', 'SDAE'),loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)
plt.tight_layout()
plt.show()
