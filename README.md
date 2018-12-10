*Project information:*
Uses neural data from experiment described in The eyes are the window to the brain (http://www.donnajobridge.com/eyes.html).

*Logistic Regression Model*
Applies sci-kit LogisticRegressionCV function and Kfold to conduct nested cross-validation. The aim is to use neural data from the hippocampus to predict the upcoming fixation. A fourier transform has been applied to the raw amplitude timeseries. Power and phase are extracted for each frequency of interest. Phase is transformed into sin and cos values to capture continuity in phase angles. Times include 750 ms leading up to the fixation and 750 ms following the fixation.

*How to run:*
To run the model, go into sklearn directory and run `python run_decoder_sklearn.py`. The specific subjects, conditions, times, and frequencies can be edited in the class input function in run_decoder_sklearn.py.
