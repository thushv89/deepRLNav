__author__ = 'thushv89'

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import pickle

combine_rviz = True
data_type = 'office' #['office','outdoor']


img_w,img_h = 640,480
rimg_w,rimg_h = 300,300
res_img_w,res_img_h = 512,384
res_rimg_w,res_rimg_h = 256,256

if combine_rviz:
    width,height = 768,480
else:
    width,height = 640,480

#[(0,7),(7,20),(53,61),(84,108),(270,284),(375,385)]
if data_type=='office':
    rviz_dir = 'test_right_pose'+os.sep+'test_rviz_office_ra_new'
    dir = "test_right_pose"+os.sep+"test_office_ra_new"
    ranges = [(7,20),(85,108),(270,284),(378,385)] #for office
    ranges_ep = [(0,50),(50,100),(250,300),(300,400)]
    exclude = [dir + os.sep + 'img_8_0.png',dir + os.sep + 'img_378_90.png'] #office
elif data_type=='outdoor':
    dir = "test_right_pose/outdoor"+os.sep+"test_outdoor_ra_new"
    rviz_dir = 'test_right_pose'+os.sep+'test_rviz_outdoor_ra_new'
    ranges = [(0,11),(192,206),(289,304)]
    ranges_ep = [(0,50),(150,200),(250,300)] # outdoor
    exclude = []
pose_type = 'pkl'

img_ep_ids = []

episode_dict = {}
if pose_type=='pkl':
    pkl_names = [f for f in listdir(dir) if isfile(join(dir, f)) and '.pkl' in f]
    max_pkl_idx = 0
    for fn in pkl_names:
        pkl_idx = int(fn.split('.')[0].split('_')[1])
        if pkl_idx>max_pkl_idx:
            max_pkl_idx=pkl_idx

    for i in range(max_pkl_idx):
        if data_type=='outdoor' and i==372:
            continue
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
    if data_type=='outdoor' and k==372:
        continue
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
    id_int,val_int = int(id_str),int(val_str)
    if id_int in file_name_dict:
        file_name_dict[id_int].append(val_int)
    else:
        file_name_dict[id_int]=[val_int]

img_idx_projected_data = {}
for i in range(np.max(file_name_dict.keys())):
    if data_type=='outdoor' and i==372:
        continue
    if i==0:
        img_idx_projected_data[i]=(episode_dict[i],-1,None)
    else:
        if i in zero_ep_entries:
            img_idx_projected_data[i]=(episode_dict[i],-1,None)
        else:
            img_idx_projected_data[i]=(episode_dict[i],action_dict[episode_dict[i]],prob_dict[episode_dict[i]])

fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
video = cv2.VideoWriter('video-'+data_type+'-ra.avi',fourcc,3,(width,height),1)

rect_h = 100
rect_w = 40
rect_x0 = 350
padding_w = 25

def compose_combined_frame(fname,rname,f_id):
    global img, p
    img = cv2.imread(fname)
    img_resized = cv2.resize(img,(res_img_w,res_img_h))
    extended_img = np.append(np.zeros((img_h-res_img_h,res_img_w,3),dtype='uint8'),img_resized,axis=0)

    img2 = cv2.imread(rname)
    if img2 is not None:
        img2_resized = cv2.resize(img2,(res_rimg_w,res_rimg_h))
        img2_extended = np.append(np.zeros((img_h-res_rimg_h,res_rimg_w,3),dtype='uint8'),img2_resized,axis=0)
    else:
        img2_extended = np.zeros((img_h,res_rimg_w,3),dtype='uint8')

    comb_img = np.asarray(np.concatenate((extended_img, img2_extended), axis=1),dtype='uint8')

    #title
    if data_type=='office':
        cv2.putText(comb_img,'Office - RA-DAE - 400 Episodes',(25,50),cv2.FONT_HERSHEY_PLAIN, 1.5,(255,255,255),2)
    elif data_type=='outdoor':
        cv2.putText(comb_img,'Outdoor - RA-DAE - 400 Episodes',(25,50),cv2.FONT_HERSHEY_PLAIN, 1.5,(255,255,255),2)
    if not fname_display is None:
        cv2.putText(comb_img, fname_display, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    cv2.putText(comb_img, episode_display, (520, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    cv2.putText(comb_img, action_display, (520, 160), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    if img_idx_projected_data[f_id][1]!=-1:
        action_prob_display = 'P('+action_str+'): %.2f'%(img_idx_projected_data[f_id][2][img_idx_projected_data[i][1]])
        cv2.putText(comb_img, action_prob_display, (520, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    if img_idx_projected_data[f_id][2] is not None:
        probs_display = 'Probs:%.2f,%.2f,%.2f' % (
        img_idx_projected_data[f_id][2][0], img_idx_projected_data[f_id][2][1], img_idx_projected_data[f_id][2][2])
        cv2.putText(comb_img, probs_display, (200, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        rect_x0_prob = 280
        for p in img_idx_projected_data[i][2]:
            rect_y1_prob = int(rect_h * p)
            if p < 0.45:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(comb_img, (rect_x0_prob, 420), (rect_x0_prob + rect_w, 420 - rect_y1_prob), color, -1)
            rect_x0_prob = rect_x0_prob + rect_w + padding_w
    return comb_img

def compose_frame(fname):
    global img, p
    img = cv2.imread(fname)

    if not fname_display is None:
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


for i in range(np.max(file_name_dict.keys())):
    if data_type=='outdoor' and i==372:
        continue
    if ranges is not None:
        in_range = [True if r[0]<=img_idx_projected_data[i][0]<=r[1] else False for r in ranges]
        if not np.any(in_range):
            continue
        else:
            r_id = np.where(np.asarray(in_range)==True)[0]
            img_ep_ids.append(i)

    while len(file_name_dict[i])>0:

        fname = dir+os.sep+'img_'+str(i)+'_'+str(np.min(file_name_dict[i]))+'.png'
        if combine_rviz:
            rname = rviz_dir+os.sep+'rviz_'+str(i)+'_'+str(np.min(file_name_dict[i]))+'.png'
        if fname not in exclude:

            print "Processing frame:%s "%(fname)

            if ranges is not None:
                fname_display = None
                episode_display = 'Episode: '+str(ranges_ep[r_id][0])+'-'+str(ranges_ep[r_id][1])
            else:
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

            if combine_rviz:
                img = compose_combined_frame(fname,rname,i)
            else:
                img = compose_frame(fname)
            video.write(img)
            if len(file_name_dict[i])==1:
                for _ in range(1):
                    if combine_rviz:
                        img = compose_combined_frame(fname,rname,i)#print "Writing Video ..."
                    else:
                        img = compose_frame(fname)
                    video.write(img)

        file_name_dict[i].remove(np.min(file_name_dict[i]))

print "Releasing Video ..."
print img_ep_ids
video.release()
cv2.destroyAllWindows()

