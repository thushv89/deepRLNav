import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

data_dir = 'image_probs'
images_per_action = 3

offic_data_dir = data_dir + os.sep + 'office'
offic_prob_filename = offic_data_dir + os.sep + 'probability_log.log'
office_image_range = [295,393]

image_selection = 'defined' # defined or random
office_misclassified = []

outdoor_data_dir = data_dir + os.sep + 'outdoor'
outdoor_prob_filename = outdoor_data_dir + os.sep + 'probability_log.log'
outdoor_image_range = [270,385]
outdoor_misclassified = []

office_image_action_map,outdoor_image_action_map = {},{}

with open(offic_prob_filename, 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',')
    for row in bumpreader:
        office_image_action_map[int(row[0])]=int(row[1])

with open(outdoor_prob_filename, 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',')
    for row in bumpreader:
        outdoor_image_action_map[int(row[0])]=int(row[1])

office_ids = [[],[],[]]
outdoor_ids = [[],[],[]]

if image_selection=='random':
    while len(office_ids[0])<images_per_action or \
        len(office_ids[1])<images_per_action or len(office_ids[2])<images_per_action:
        rand_key = np.random.randint(office_image_range[0],office_image_range[1])
        curr_act =  office_image_action_map[rand_key]
        if len(office_ids[curr_act])<images_per_action and rand_key not in office_ids[curr_act]:
            office_ids[curr_act].append(rand_key)
        else:
            continue

    while len(outdoor_ids[0])<images_per_action or \
        len(outdoor_ids[1])<images_per_action or len(outdoor_ids[2])<images_per_action:
        rand_key = np.random.randint(outdoor_image_range[0],outdoor_image_range[1])
        curr_act =  outdoor_image_action_map[rand_key]
        if len(outdoor_ids[curr_act])<images_per_action and rand_key not in outdoor_ids[curr_act]:
            outdoor_ids[curr_act].append(rand_key)
        else:
            continue
elif image_selection=='defined':

    office_ids = [[365,377,342],[369,354,357],[367,360,347]]
    outdoor_ids = [[319,339,279],[321,368,338],[348,356,353]]
    misclassified_ids = [335,339,348,273,317,315]
print('========= Office ========')
print('0: ',office_ids[0])
print('1: ',office_ids[1])
print('2: ',office_ids[2])
print('========= Outdoors ========')
print('0: ',outdoor_ids[0])
print('1: ',outdoor_ids[1])
print('2: ',outdoor_ids[2])

if image_selection=='random':
    fig, axarr = plt.subplots(3,2*images_per_action)
elif image_selection=='defined':
    fig, axarr = plt.subplots(4,2*images_per_action)

actions_strings = ['Right','Straight','Left']

for r_i in range(0,3):
    for c_i in range(2):
        if c_i==0:
            for i in range(images_per_action):
                img =  mpimg.imread(offic_data_dir+os.sep+'img_'+str(office_ids[r_i][i])+'_0.png')
                axarr[r_i][c_i*images_per_action+i].imshow(img)
                axarr[r_i][c_i*images_per_action+i].axis('off')
            axarr[r_i][c_i*images_per_action+i].set_title('Images classified action as '+actions_strings[r_i])
        if c_i==1:
            for i in range(images_per_action):
                img =  mpimg.imread(outdoor_data_dir+os.sep+'img_'+str(outdoor_ids[r_i][i])+'_0.png')
                axarr[r_i][c_i*images_per_action+i].imshow(img)
                axarr[r_i][c_i*images_per_action+i].axis('off')

for c_i in range(2*images_per_action):
    if c_i<images_per_action:
        img =  mpimg.imread(offic_data_dir+os.sep+'img_'+str(misclassified_ids[c_i])+'_0.png')
        axarr[3][c_i].imshow(img)
        axarr[3][c_i].axis('off')
    else:
        img =  mpimg.imread(outdoor_data_dir+os.sep+'img_'+str(misclassified_ids[c_i])+'_0.png')
        axarr[3][c_i].imshow(img)
        axarr[3][c_i].axis('off')

fig.subplots_adjust(wspace=0.05,hspace=0.15,bottom=0.01,top=0.95,right=0.99,left=0.01)
#plt.tight_layout()
plt.show()