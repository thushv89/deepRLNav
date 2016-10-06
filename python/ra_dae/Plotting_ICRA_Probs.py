import csv
import numpy as np

filenames = ['probability_log_lr.log','probability_log_sd.log','probability_log_office_lr.log','probability_log_office_sd.log','probability_log_office_ra.log','probability_log_outdoor_lr.log','probability_log_outdoor_sd.log','probability_log_outdoor_ra.log']
algo = ['SIM LR','SIM SD','Office LR','Office SD','Office RA','Outdoor LR','Outdoor SD','Outdoor RA']
ra_probs = []
sd_probs = []

def get_file_data(fn):
    episodes=[]
    actions=[]
    probs=[]
    with open(fn, 'rt') as csvfile:
        bumpreader = csv.reader(csvfile, delimiter=',')
        for row in bumpreader:
            temp_probs = []
            episodes.append(int(row[0]))
            actions.append(int(row[1]))
            prob_tokens = row[2][1:len(row[2])-1].split(' ')
            for p in prob_tokens:
                if len(p)>2:
                    temp_probs.append(float(p))
            probs.append(temp_probs)

    return episodes,actions,probs


def print_info(algo,eps,a,p):
    exc_action =[]
    for i in range(len(a)):
        exc_action.append(p[a[i]])

    print("%s:%.2f$\pm$%.2f"%(algo,np.mean(exc_action),np.std(exc_action)))

for j,fn in enumerate(filenames):
    eps,a,p = get_file_data(fn)
    print_info(algo[j],eps,a,p)