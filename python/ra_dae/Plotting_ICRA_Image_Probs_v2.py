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
office_image_range = [294,393]

outdoor_data_dir = data_dir + os.sep + 'outdoor'
outdoor_prob_filename = outdoor_data_dir + os.sep + 'probability_log.log'
outdoor_image_range = [270,385]

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


fig, axarr = plt.subplots(3,2*images_per_action)

for r_i in range(0,3):
    for c_i in range(2):
        if c_i==0:
            for i in range(images_per_action):
                print(r_i,',',c_i,',',i)
                img =  mpimg.imread(offic_data_dir+os.sep+'img_'+str(office_ids[r_i][i])+'_0.png')
                axarr[r_i][c_i*images_per_action+i].imshow(img)
                axarr[r_i][c_i*images_per_action+i].axis('off')
        if c_i==1:
            for i in range(images_per_action):
                print(r_i,',',c_i,',',i)
                img =  mpimg.imread(outdoor_data_dir+os.sep+'img_'+str(outdoor_ids[r_i][i])+'_0.png')
                axarr[r_i][c_i*images_per_action+i].imshow(img)
                axarr[r_i][c_i*images_per_action+i].axis('off')

fig.subplots_adjust(wspace=0.05,hspace=0.05,bottom=0.01,top=0.99,right=0.99,left=0.01)
#plt.tight_layout()
plt.show()