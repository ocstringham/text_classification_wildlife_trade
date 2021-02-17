# script to run sensitivity analysis (reduce training set size)

import pandas as pd
import numpy as np

# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

from functions.run_text_classifier_get_metrics import run_text_classifier_get_metrics



# define classifier
classifier =  LogisticRegression(max_iter = 5000)
            
# define CV folds and iterations    
n_folds = 10
n_iter = 100 # number of stochastic iterations


# get training set sizes
df = pd.read_csv('../data/raw/text_w_labels.csv', encoding = "ISO-8859-1")
label1_size = df[df['label1'].notnull()].shape[0] * (0.9) # 0.9 because 10-fold CV
label2_size = df[df['label2'].notnull()].shape[0] * (0.9)
label3_size = df[df['label3'].notnull()].shape[0] * (0.9)

# define label names
labels = ['label1','label2','label3']

# name of col with text in it
text_col = 'text_stop_stem'


# create list of diminishing sample sizes
amnts = np.arange(500, 15000, 500).tolist() 


# inits for storing data
df_metrics_list = []
# df_roc_pr_list = []
# df_coeffs_list = []


# loop over labels and run sensitivity analysis
for i1, label in enumerate(labels):

    ## assign training set size by label
    if label == "label1":
        size_temp = label1_size
    elif label == "label2":
        size_temp = label2_size
    elif label == "label3":
        size_temp = label3_size
    else:
        size_temp = np.nan

    # loop over amounts to decrease training set by
    for i2, amnt in enumerate(amnts):
        
        # only run if dims are not negative
        if amnt < size_temp: 
        
            # loop over iterations
            for i3 in range(0, n_iter):
                    
                # run CV text classification
                temp, junk1, junk2 = run_text_classifier_get_metrics(classifier, 
                                             n_folds, df, text_col, label,
                                             decrease_n = True,
                                             decrease_amnt = amnt)
                
                # assign iteration
                temp['iteration'] = i3
                
                ## save
                df_metrics_list.append(temp)
                # df_roc_pr_list.append(temp[1])
                # df_coeffs_list.append(temp[2])
                
                print('iteration {} of {}'.format(i3+1, n_iter))
            
        ## print update
        print("Finished amount {} of {} for Label {} of {}"\
              .format(i2 + 1, len(amnts), i1 + 1, len(labels)))
            
            
# unlist data
df_metrics = pd.concat(df_metrics_list)
# df_roc_pr = pd.concat(df_roc_pr_list)
# df_coeffs = pd.concat(df_coeffs_list)

# save
df_metrics.to_csv('../data/sensitivity_analysis_cv_metrics.csv' )
# df_roc_pr.to_csv('../data/sensitivity_analysis_cv_roc_pr.csv')
# df_coeffs.to_csv('../data/sensitivity_analysis/cv_coeffs.csv')
