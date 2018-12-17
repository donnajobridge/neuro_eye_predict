import numpy as np
import pandas as pd
import os.path
from sklearn.model_selection import KFold, train_test_split
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
                 label_key=None,
                 freq_range=None,
                 time_range=None,
                 dsrate=None,
                 freqs = pd.DataFrame(np.logspace(np.log10(3),np.log10(100),num=50), index=np.arange(1,51), columns=['freqs']),
                 times = pd.DataFrame(np.arange(-750,751,step=2),index=np.arange(1,752), columns=['time']),
                 keep_pow=True,
                 keep_phase=True):
        self.sub = sub
        self.cond=cond
        self.label_key = label_key
        self.data_path=data_path
        self.data_file = [f'{self.data_path}{self.cond}_behav_{sub}.csv']

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

        self.predcv = self.data_path+self.sub+'_'+self.cond+'_'+self.label_key+'_'+self.features'ridge_predictions.csv'

        self.scores=self.data_path+self.sub+'_'+self.cond+'_'+self.label_key+'_'+ self.features+'ridge_scores.csv'
        # parameters for cross-validation

        self.nperms = 10

        # parameters for sklearn inputs
        self.freqs=freqs
        self.freq_inds = [freqs[(freqs['freqs'] >= freq_range[count][0]) & (freqs['freqs'] <= freq_range[count][1])].index.values for count in range(len(freq_range))]
        self.times=times
        self.time_inds = [times[(times['time'] >= time_range[count][0]) & (times['time'] <= time_range[count][1])].index.values for count in range(len(time_range))]


    def perform_ridgecv(self, permute_flag=True):
        '''perform nested cross validation using LogisticRegressionCV
        loops through each frequency & time range for each model_selection
        outer loop performs kfold split; inner loop searches for best C parameter (using kfold splits) and fits it to withheld test data from the outer loop kfold
        '''
        df = self.make_df()
        df_norm = normalize_data(df, self.keep_pow, self.keep_phase)
        df_ds = downsample_timeseries(df_norm, self.dsrate)

        for f in self.freq_inds:
            for t in self.time_inds:
                print('beginning time:',t,'beginning freq:',f)

                scorelist = []
                cond_df = pd.DataFrame()

                df = gen_wide_df(df, f, t)

                # get data & labels
                X,y = get_data_and_labels(df, self.label_key, df.index.get_values())
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                ridgecv = initialize_ridgecv(X_train, y_train)
                # dictionary of values with probabilities from ridgecv
                probsdict = {'preds':ridgecv.predict(X_test), 'labels':y_test.reset_index(drop=True), 'orig_ind':test_ind, 'start_time':self.times.loc[t[0], 'time'], 'end_time':self.times.loc[t[-1], 'time'], 'start_freq':self.freqs.loc[f[0], 'freqs'], 'end_freq':self.freqs.loc[f[-1],'freqs'], 'features':self.features, 'dsrate':self.dsrate}

                cond_df = pd.DataFrame(probsdict)

                if os.path.isfile(self.predcv):
                    cond_df.to_csv(self.predcv, mode='a', header=False)
                else:
                    cond_df.to_csv(self.predcv)

                real_mse = mean_squared_error(cond_df['labels'], cond_df['preds'])

                # permute class labels and run model to find a null distribution of auc values
                if permute_flag:
                    mse_z, fake_mses = run_perm_test(df, real_mse, self.nperms, self.label_key)
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
        df['class']=1
        cols_to_drop = [f'view{x}' for x in range(1,4)]+['response']
        # subtract loc3 viewing from location of interest
        df[f'{self.label_key}-view3'] = df[self.label_key] - df['view3']
        df.drop(cols_to_drop, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def initialize_ridgecv(X_train, y_train):
    ''' sets up model to run ridge regression cross-validation'''
    alphas = np.array([10,50,100])
    ridgecv = RidgeCV(alphas=alphas, cv=None, scoring ='neg_mean_squared_error')
    ridgecv.fit(X_train, y_train)
    return ridgecv

def run_perm_test(df, real_mse, nperms, label_key):
    ''' creates model using false class labels, finds mse score n times to create null distribution '''
    fake_mses = pd.DataFrame()
    for i in range(0,nperms):

        # get permuted data
        X,y = get_data_and_labels(df, label_key, df.index.get_values(), permute=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        ridgecv = initialize_ridgecv(X_train, y_train)

        predict_perm = pd.DataFrame(ridgecv.predict(X_test))
        predict_perm['labels'] = y_test.reset_index(drop=True)

        fake_mses.loc[i,0] = mean_squared_error(predict_perm['labels'],predict_perm[0])
    mse_z = ((real_mse - fake_mses.mean())/fake_mses.std()).get_values()
    plot_null(fake_mses)

    return mse_z, fake_mses

def plot_null(fake_aucs):
    print(fake_aucs)
    plt.hist(fake_aucs[0])
    plt.show()
