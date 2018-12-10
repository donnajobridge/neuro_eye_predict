
from location_model_sklearn import Decoder

for subject in ['ec109']:
# for subject in ['ec105', 'ec106', 'ec107', 'ec108', 'ec109']:
    cond1 = ['mismatch_old']#, 'mismatch_new']
    cond2 = 'match_old'
    cond_list = []
    for c1 in cond1:
        cond_dict = {}
        cond_dict['cond1'] = c1
        cond_dict['cond2'] = cond2
        cond_list.append(cond_dict)

    for cond_pair in cond_list:
        decoder = Decoder(sub = subject,
                                cond1 = cond_pair['cond1'],
                                cond2 = cond_pair['cond2'],
                                data_path='/Users/drdj/neuro_eye_predict/data/',
                                freq_range = [(3,8), (8,15), (15,25), (25,60), (60,101)],
                                time_range = [(-750,-500), (-500,-250), (-250,0), (0,250), (250,500), (500,750)])

        decoder.perform_cv()
