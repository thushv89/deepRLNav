__author__ = 'thushv89'

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import pickle


dir = "test_right_pose"+os.sep+"test_office_ra_new"
width,height = 640,480
pose_type = 'pkl'

episode_dict = {}
if pose_type=='pkl':
    pkl_names = [f for f in listdir(dir) if isfile(join(dir, f)) and '.pkl' in f]
    max_pkl_idx = 0
    for fn in pkl_names:
        pkl_idx = int(fn.split('.')[0].split('_')[1])
        if pkl_idx>max_pkl_idx:
            max_pkl_idx=pkl_idx

    for i in range(max_pkl_idx):

        with open(dir+os.sep+'pose_'+str(i)+'.pkl', 'rb') as f:
            data = pickle.load(f)
            for point in data:
                episode_dict[int(point[0])] = int(point[2])

elif pose_type=='log':

    with open(dir+'/trajectory.log') as f:
        f = f.readlines()

    for line in f:
        string_tokens = line.split(',')
        if len(string_tokens)<10:
            continue

        img_ep = int(string_tokens[0])
        if img_ep in episode_dict:
            continue
        algo_ep = int(string_tokens[2])
        episode_dict[img_ep] = algo_ep

#correct 0 episode entries for each restart
zero_ep_entries = []
for k in range(np.max(episode_dict.keys())):
    if k!=0:
        if episode_dict[k]==0:
            episode_dict[k]=episode_dict[k-1]+1
            zero_ep_entries.append(k)
        else:
            episode_dict[k] = episode_dict[k]-1


with open(dir+'/probability_log.log') as f:
    f = f.readlines()

prob_dict = {}
action_dict = {}
for line in f:
    string_tokens = line.split(',')
    algo_ep = int(string_tokens[0])
    action = int(string_tokens[1])
    probs_str = string_tokens[2][1:-2].split(' ')
    probs = []
    action_dict[algo_ep]=action
    for p in probs_str:
        if len(p)>1:
            probs.append(float(p))
    prob_dict[algo_ep] = probs

img_names = [f for f in listdir(dir) if isfile(join(dir, f)) and '.png' in f]
file_name_dict = {}

for name in img_names:
    id_str,val_str = name[4:-4].split('_')
    print id_str,val_str
    id_int,val_int = int(id_str),int(val_str)
    if id_int in file_name_dict:
        file_name_dict[id_int].append(val_int)
    else:
        file_name_dict[id_int]=[val_int]

img_idx_projected_data = {}
for i in range(np.max(file_name_dict.keys())):
    if i==0:
        img_idx_projected_data[i]=(episode_dict[i],-1,None)
    else:
        if i in zero_ep_entries:
            img_idx_projected_data[i]=(episode_dict[i],-1,None)
        else:
            img_idx_projected_data[i]=(episode_dict[i],action_dict[episode_dict[i]],prob_dict[episode_dict[i]])

fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('new-video.avi',fourcc,3,(width,height),1)

rect_h = 100
rect_w = 40
rect_x0 = 350
padding_w = 25

def compose_frame(fname):
    global img, p
    img = cv2.imread(fname)

    cv2.putText(img, fname_display, (50, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    cv2.putText(img, episode_display, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    cv2.putText(img, action_display, (50, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    if img_idx_projected_data[i][2] is not None:
        probs_display = 'Probs:%.2f,%.2f,%.2f' % (
        img_idx_projected_data[i][2][0], img_idx_projected_data[i][2][1], img_idx_projected_data[i][2][2])
        cv2.putText(img, probs_display, (300, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        rect_x0_prob = 380
        for p in img_idx_projected_data[i][2]:
            rect_y1_prob = int(rect_h * p)
            if p < 0.45:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(img, (rect_x0_prob, 420), (rect_x0_prob + rect_w, 420 - rect_y1_prob), color, -1)
            rect_x0_prob = rect_x0_prob + rect_w + padding_w

    return img

for i in range(np.max(file_name_dict.keys())//10):
    while len(file_name_dict[i])>0:

        fname = dir+os.sep+'img_'+str(i)+'_'+str(np.min(file_name_dict[i]))+'.png'

        print "Processing frame:%s "%(fname)
        fname_display = 'image-'+str(i)+'-'+str(np.min(file_name_dict[i]))
        episode_display = 'Episode: '+str(img_idx_projected_data[i][0])
        if img_idx_projected_data[i][1]==0:
            action_str = 'Left'
        elif img_idx_projected_data[i][1]==1:
            action_str = 'Straight'
        elif img_idx_projected_data[i][1]==2:
            action_str = 'Right'
        elif img_idx_projected_data[i][1]==-1:
            action_str = 'Initial Step'

        action_display = 'Action:'+action_str

        img = compose_frame(fname)
        video.write(img)
        if len(file_name_dict[i])==1:
            for _ in range(3):
                img = compose_frame(fname)#print "Writing Video ..."
                video.write(img)

        file_name_dict[i].remove(np.min(file_name_dict[i]))

print "Releasing Video ..."
video.release()
cv2.destroyAllWindows()

