__author__ = 'thushv89'

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

dir = "test_outdoor_ra"
width,height = 640,480

with open(dir+'/trajectory.log') as f:
    f = f.readlines()

episode_dict = {}
for line in f:
    string_tokens = line.split(',')
    if len(string_tokens)<10:
        continue

    img_ep = int(string_tokens[0])
    if img_ep in episode_dict:
        continue
    algo_ep = int(string_tokens[2])
    episode_dict[img_ep] = algo_ep

with open(dir+'/probability_log.log') as f:
    f = f.readlines()

prob_dict = {}
for line in f:
    string_tokens = line.split(',')
    algo_ep = int(string_tokens[0])
    action = int(string_tokens[1])
    probs_str = string_tokens[2][1:-2].split(' ')
    probs = []
    for p in probs_str:
        if len(p)>1:
            probs.append(float(p))
    prob_dict[algo_ep] = probs

img_names = [f for f in listdir(dir) if isfile(join(dir, f)) and '.png' in f]
file_name_dict = {}

for name in img_names:
    id_str,val_str = name[3:-4].split('_')
    print id_str,val_str
    id_int,val_int = int(id_str),int(val_str)
    if id_int in file_name_dict:
        file_name_dict[id_int].append(val_int)
    else:
        file_name_dict[id_int]=[val_int]

fourcc = cv2.cv.CV_FOURCC(*'XVID')
video = cv2.VideoWriter('video.avi',fourcc,3,(width,height),1)

rect_h = 100
rect_w = 40
rect_x0 = 350
padding_w = 25
for i in range(np.max(file_name_dict.keys())//10):
    while len(file_name_dict[i])>0:
        fname = dir+'/img'+str(i)+'_'+str(np.min(file_name_dict[i]))+'.png'
        fname_display = 'image-'+str(i)+'-'+str(np.min(file_name_dict[i]))
        episode_display = 'Episode: '+str(episode_dict[i])
        algo_ep = episode_dict[i]
        probs_display = 'Probs:%.2f,%.2f,%.2f'%(prob_dict[algo_ep][0],prob_dict[algo_ep][1],prob_dict[algo_ep][2])
        img = cv2.imread(fname)
        file_name_dict[i].remove(np.min(file_name_dict[i]))
        cv2.putText(img, fname_display,(50, 50),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
        cv2.putText(img, episode_display,(50, 75),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
        cv2.putText(img, probs_display,(300, 460),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)

        rect_x0_prob = 380
        for p in prob_dict[algo_ep]:
            rect_y1_prob = int(rect_h*p)
            if p<0.45:
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(img, (rect_x0_prob ,420), (rect_x0_prob+rect_w,420-rect_y1_prob), color, -1)
            rect_x0_prob = rect_x0_prob + rect_w + padding_w

        video.write(img)

video.release()
cv2.destroyAllWindows()

