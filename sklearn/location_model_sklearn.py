import numpy as np
import pandas as pd
import os.path
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class Decoder(object):
    """
    This class constructs a decoder that will learn to map from multivariate neural data to the location of a fixation

    Parameters
    ---
    data_files : str
        Location of where the neural data is to be loaded (list of files per class)
    """

    def __init__(self,
                 sub=None,
                 cond1=None,
                 cond2=None,
                 data_path=None,
                 long_csv=None,
                 label_key='class',
                 freq_range=None,
                 time_range=None,
                 freqs = pd.DataFrame(np.logspace(np.log10(3),np.log10(100),num=50), index=np.arange(1,51), columns=['freqs']),
                 times = pd.DataFrame(np.arange(-750,751,step=2),index=np.arange(1,752), columns=['time'])):
        self.sub = sub
        self.cond1=cond1
        self.cond2=cond2
        self.data_path=data_path
        self.data_files = [f'{self.data_path}{self.cond1}_behav_{sub}.csv',
                           f'{self.data_path}{self.cond2}_behav_{sub}.csv']
        # self.long_csv = f'{self.data_path}{self.sub}_{self.cond1}_{self.cond2}_long.csv'

        # check if the df exists, if not create it
        # self.gen_long()
        # parameters for cross-validation

        self.nfold = 5
        self.nperms = 500

        # parameters for sklearn inputs
        self.label_key = label_key
        self.freqs=freqs
        self.freq_inds = [freqs[(freqs['freqs'] >= freq_range[count][0]) & (freqs['freqs'] <= freq_range[count][1])].index.values for count in range(len(freq_range))]
        self.times=times
        self.time_inds = [times[(times['time'] >= time_range[count][0]) & (times['time'] <= time_range[count][1])].index.values for count in range(len(time_range))]

        self.pred_cv = f'{self.data_path}{self.sub}_{self.cond1}_{self.cond2}_preds.csv'

    def perform_cv(self, permute_flag=True):
        kf = KFold(n_splits=self.nfold, shuffle=True)
        df_long = self.gen_long()
        scorelist = []
        pred_df = pd.DataFrame()

        for f in self.freq_inds:
            for t in self.time_inds:
                cond_df = pd.DataFrame()
                df, elecs, freqs, time, ncol = gen_wide_df(df_long, f, t)
                df.reset_index(drop=False, inplace=True)
                # iterate over folds
                for train_ind, test_ind in kf.split(df):

                    # data for this fold
                    X_train,y_train = get_data_and_labels(df, self.label_key, train_ind)
                    X_test,y_test = get_data_and_labels(df, self.label_key, test_ind)

                    logregcv = initialize_logreg(X_train, y_train, self.nfold)

                    probsdict = {'probs':pd.DataFrame(logregcv.predict_proba(X_test)).loc[:,1], 'labels':y_test.reset_index(drop=True), 'orig_ind':test_ind, 'start_time':self.times.loc[t[0], 'time'], 'end_time':self.times.loc[t[-1], 'time'], 'start_freq':self.freqs.loc[f[0], 'freqs'], 'end_freq':self.freqs.loc[f[-1],'freqs']}

                    fold_df = pd.DataFrame(probsdict)
                    cond_df = pd.concat([cond_df, fold_df])
                    pred_df = pd.concat([pred_df, fold_df])
                real_auc = roc_auc_score(cond_df['labels'], cond_df['probs'])

                if permute_flag:
                    auc_z, fake_aucs = run_perm_test(df, kf, real_auc, self.nperms, self.label_key, self.nfold)
                    p = 1-((np.sum(real_auc>fake_aucs[0])+1)/(fake_aucs.shape[0]+1))

                    scoredict = {'sub':self.sub, 'starttime':self.times.loc[t[0], 'time'], 'endtime':self.times.loc[t[-1], 'time'], 'startfreq':self.freqs.loc[f[0], 'freqs'], 'endfreq':self.freqs.loc[f[-1], 'freqs'], 'real_auc':real_auc, 'auc_z':auc_z, 'pvalue':p, 'nperms':self.nperms}
                    scorelist.append(scoredict)
                    print(f,t,scorelist)
        if permute_flag:
            scoredf = pd.DataFrame(scorelist)
            scoredf.to_csv(self.data_path+self.sub+'_'+self.cond1+'_'+self.cond2+'_'+'scores.csv')

        pred_df.to_csv(self.data_path+self.sub+'_'+self.cond1+'_'+self.cond2+'_'+'predictions.csv')


    def gen_long(self):
        """
        Generates a long format CSV
        :return:
        """

        # read header on first
        df = pd.read_csv(self.data_files[0])
        df['class'] = 0

        for i, file in enumerate(self.data_files[1:]):
            df_tmp = pd.read_csv(file)
            df_tmp['class'] = i + 1
            df = pd.concat([df, df_tmp], axis=0)

        df.reset_index(drop=True, inplace=True)
        return df
        # df.to_csv(self.long_csv, index=False)


def gen_wide_df(df, f_idx, t_idx):
    """

    :param df:
    :param f_idx:
    :param t_idx:
    :return:
    """
    cols_to_drop = ['response', 'view1', 'view2', 'view3']
    df.drop(cols_to_drop, axis=1, inplace=True)

    df = df.loc[(df['time'].isin(t_idx)) & (df['freqs'].isin(f_idx))]

    # z-score within block
    gb = df.groupby(['elecs', 'freqs', 'blockno'])
    df['zpow'] = gb['pow'].apply(zscore)
    df['sin'] = gb['phase'].apply(np.sin)
    df['cos'] = gb['phase'].apply(np.cos)
    df.drop(['pow', 'phase'], inplace=True, axis=1)

    df = pd.pivot_table(df,
                        index=['class', 'blockno', 'events'],
                        columns=['elecs', 'freqs', 'time'],
                        aggfunc=[lambda x: x])

    elecs = df.columns.get_level_values('elecs')
    freqs = df.columns.get_level_values('freqs')
    time = df.columns.get_level_values('time')
    ncol = len(elecs)

    return df, elecs, freqs, time, ncol


def get_data_and_labels(df, label_key, index):

    y = df.loc[index, label_key]
    X = df.loc[index, :].drop(label_key, axis=1)

    return X,y

def get_data_and_labels_permuted(df, label_key, index):

    df[label_key] = np.random.permutation(df[label_key])
    y = df.loc[index, label_key]
    X = df.loc[index, :].drop(label_key, axis=1)

    return X,y


def zscore(x):
    z = (x - np.mean(x)) / np.std(x)
    return z


def initialize_logreg(X_train, y_train, nfolds):
    C_grid = np.logspace(-15,-1,10)
    logregcv = LogisticRegressionCV(Cs=C_grid, cv=nfolds, max_iter=1000,
                                    scoring ='roc_auc',class_weight = 'balanced', n_jobs=-1,penalty='l2', solver='lbfgs')
    logregcv.fit(X_train, y_train)
    return logregcv

def run_perm_test(df, kf, real_auc, nperms, label_key, nfolds):
    fake_aucs = pd.DataFrame()
    for i in range(0,nperms):
        all_preds = None
        for train_ind, test_ind in kf.split(df):

            # data for this fold
            X_train,y_train = get_data_and_labels_permuted(df, label_key, train_ind)
            X_test,y_test = get_data_and_labels_permuted(df, label_key, test_ind)

            logregcv = initialize_logreg(X_train, y_train, nfolds)

            predict_fold = pd.DataFrame(logregcv.predict_proba(X_test))
            predict_fold['labels'] = y_test.reset_index(drop=True)
            if all_preds is None:
                all_preds = predict_fold.copy()
            else:
                all_preds = pd.concat([all_preds, predict_fold])
        fake_aucs.loc[i,0] = roc_auc_score(all_preds['labels'],all_preds[1])

    auc_z = (real_auc - fake_aucs.mean())/fake_aucs.std()
    return auc_z, fake_aucs

def plot_null(fake_aucs):
    print(fake_aucs)
    plt.hist(fake_aucs[0])
    plt.show()
