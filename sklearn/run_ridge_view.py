
from ridge_predict_view import ridge

conds = ['mismatch_old', 'mismatch_new']
for subject in ['ec109']:
# for subject in ['ec105', 'ec106', 'ec107', 'ec108', 'ec109']:
    for condition in conds:
        ridge = ridge(sub = subject,
                                cond = condition,
                                predictor = 'view1',
                                data_path='/Users/drdj/neuro_eye_predict/data/',
                                freq_range = [(3,8)],# (8,15), (15,25), (25,60), (60,101)],
                                # time_range = [(-750,-500), (-500,-250), (-250,0), (0,250), (250,500), (500,750)],
                                dsrate = 10,
                                keep_pow = False,
                                keep_phase = True)
        ridge.perform_ridgecv()
