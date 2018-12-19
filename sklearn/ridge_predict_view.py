import numpy as np
import pandas as pd
import os.path
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from location_model_sklearn import normalize_data, downsample_timeseries, gen_wide_df, get_data_and_labels

class ridge(object):
    """
    This class constructs a decoder that will learn to map from multivariate neural data to the location of a fixation

    Parameters
    ---
    data_files : str
        Location of where the neural data is to be loaded (list of files per class)
    cond : str
        Condition name
    label_key: variable to predict
    freq_range:   list of tuples
        Input lower and upper bounds of frequencies to include in each model
    time_range: list of tuples
        Input lower and upper bounds of times to include in each model
    """

    def __init__(self,
                 sub=None,
                 cond=None,
                 data_path=None,
                 long_csv=None,
                 predictor=None,
                 freq_range=None,
                 time_range=None,
                 dsrate=1,
                 freqs = pd.DataFrame(np.logspace(np.log10(3),np.log10(100),num=50), index=np.arange(1,51), columns=['freqs']),
                 keep_pow=True,
                 keep_phase=True):
        self.sub = sub
        self.cond=cond
        self.predictor = predictor
        self.label_key = 'class'
        self.data_path=data_path
        self.data_file = f'{self.data_path}{self.cond}_behav_{sub}.csv'

        # specify features to use
        self.keep_pow=keep_pow
        self.keep_phase=keep_phase
        self.dsrate=dsrate
        if self.keep_pow and self.keep_phase:
            self.features = 'pow+phase'
        elif self.keep_pow and not self.keep_phase:
            self.features = 'pow'
        elif not self.keep_pow and self.keep_phase:
            self.features= 'phase'

        self.predcv = self.data_path+self.sub+'_'+self.cond+'_'+self.predictor+'_'+self.features+'_ridge_predictions.csv'

        self.scores=self.data_path+self.sub+'_'+self.cond+'_'+self.predictor+'_'+ self.features+'_ridge_scores.csv'
        # parameters for cross-validation

        self.nperms = 10
        self.nfold = 5
        # parameters for sklearn inputs
        self.freqs=freqs
        self.freq_inds = [freqs[(freqs['freqs'] >= freq_range[count][0]) & (freqs['freqs'] <= freq_range[count][1])].index.values for count in range(len(freq_range))]

        self.times=pd.DataFrame(np.arange(-750,751,step=dsrate*2),
        index=np.arange(1,752, step=dsrate), columns=['time'])

        if time_range:
            self.time_inds = [self.times[(self.times['time'] >= time_range[count][0]) & (self.times['time'] <= time_range[count][1])].index.values for count in range(len(time_range))]
        else:
            self.time_inds = [[x] for x in self.times.index.values]

    def perform_ridgecv(self, permute_flag=True):
        '''perform nested cross validation using LogisticRegressionCV
        loops through each frequency & time range for each model_selection
        outer loop performs kfold split; inner loop searches for best C parameter (using kfold splits) and fits it to withheld test data from the outer loop kfold
        '''
        kf = KFold(n_splits=self.nfold, shuffle=True)

        df = self.make_df()
        df_norm = normalize_data(df, self.keep_pow, self.keep_phase)
        df_ds = downsample_timeseries(df_norm, self.dsrate)

        for f in self.freq_inds:
            for t in self.time_inds:
                # print('beginning time:',t,'beginning freq:',f)

                scorelist = []
                cond_df = pd.DataFrame()

                df = gen_wide_df(df_ds, f, t)
                print(df)
                for train_ind, test_ind in kf.split(df):

                    # data for this fold
                    X_train,y_train = get_data_and_labels(df, self.label_key, train_ind)
                    X_test,y_test = get_data_and_labels(df, self.label_key, test_ind)

                    ridgecv = initialize_ridgecv(X_train, y_train, self.nfold)
                    # dictionary of values with probabilities from ridgecv
                    probsdict = {'preds':ridgecv.predict(X_test), 'labels':y_test.reset_index(drop=True), 'orig_ind':test_ind, 'start_time':self.times.loc[t[0], 'time'], 'end_time':self.times.loc[t[-1], 'time'], 'start_freq':self.freqs.loc[f[0], 'freqs'], 'end_freq':self.freqs.loc[f[-1],'freqs'], 'features':self.features, 'dsrate':self.dsrate}

                    fold_df = pd.DataFrame(probsdict)
                    cond_df = pd.concat([cond_df, fold_df])

                if os.path.isfile(self.predcv):
                    cond_df.to_csv(self.predcv, mode='a', header=False)
                else:
                    cond_df.to_csv(self.predcv)

                real_mse = mean_squared_error(cond_df['labels'], cond_df['preds'])

                # permute class labels and run model to find a null distribution of auc values
                if permute_flag:
                    mse_z, fake_mses = run_perm_test(df, kf, real_mse, self.nperms, self.label_key, self.nfold)
                    p = 1-((np.sum(real_mse>fake_mses[0])+1)/(fake_mses.shape[0]+1))
                else:
                    mse_z=None
                    p=None

                # dict of scores
                scoredict = {'sub':self.sub, 'starttime':self.times.loc[t[0], 'time'], 'endtime':self.times.loc[t[-1], 'time'], 'startfreq':self.freqs.loc[f[0], 'freqs'], 'endfreq':self.freqs.loc[f[-1], 'freqs'], 'real_mse':real_mse, 'mse_z':mse_z, 'pvalue':p, 'nperms':self.nperms, 'dsrate':self.dsrate}
                scorelist.append(scoredict)

                scoredf = pd.DataFrame(scorelist)
                if os.path.isfile(self.scores):
                    scoredf.to_csv(self.scores, mode='a', header=False)
                else:
                    scoredf.to_csv(self.scores)


    def make_df(self):
        """
        Removes extra data columns & creates df
        :return: df
        """
        # read in file
        df = pd.read_csv(self.data_file)
        cols_to_drop = [f'view{x}' for x in range(1,4)]+['response']
        # subtract loc3 viewing from location of interest
        df[self.label_key] = df[self.predictor] - df['view3']
        df.drop(cols_to_drop, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


def initialize_ridgecv(X_train, y_train, nfolds):
    ''' sets up model to run ridge regression cross-validation'''
    alphas = np.logspace(1,5, 10)
    ridgecv = RidgeCV(alphas=alphas, cv=nfolds, scoring ='neg_mean_squared_error')
    ridgecv.fit(X_train, y_train)
    return ridgecv

def run_perm_test(df, kf, real_mse, nperms, label_key, nfolds):
    ''' creates model using false class labels, finds mse score n times to create null distribution '''
    fake_mses = pd.DataFrame()
    for i in range(0,nperms):
        all_preds = None

        for train_ind, test_ind in kf.split(df):

            # data for this fold
            X_train,y_train = get_data_and_labels(df, label_key, train_ind, permute=True)
            X_test,y_test = get_data_and_labels(df, label_key, test_ind, permute=True)

            ridgecv = initialize_ridgecv(X_train, y_train, nfolds)

            predict_fold = pd.DataFrame(ridgecv.predict(X_test))
            predict_fold['labels'] = y_test.reset_index(drop=True)

            if all_preds is None:
                all_preds = predict_fold.copy()
            else:
                all_preds = pd.concat([all_preds, predict_fold])

            fake_mses.loc[i,0] = mean_squared_error(all_preds['labels'],all_preds[0])

    mse_z = ((real_mse - fake_mses.mean())/fake_mses.std()).get_values()

    plot_null(fake_mses)

    return mse_z, fake_mses

def plot_null(fake_aucs):
    print(fake_aucs)
    plt.hist(fake_aucs[0])
    plt.show()
    plt.clf()
