
filenames = ['bump_log_train_office_sd.log','bump_log_train_office_ra.log']
ra_probs = []
sd_probs = []

with open(filenames[0], 'rt') as csvfile:
    bumpreader = csv.reader(csvfile, delimiter=',')
    for row in bumpreader:
        row_tokens = string(row)
        print(row_tokens)

