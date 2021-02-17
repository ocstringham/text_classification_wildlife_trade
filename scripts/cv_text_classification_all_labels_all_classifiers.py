import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from functions.run_text_classifier_get_metrics import run_text_classifier_get_metrics


# inits for labels and classifiers

## data frame of text data
file_loc = '../data/raw/text_w_labels.csv'
df = pd.read_csv( file_loc, encoding = "ISO-8859-1")

## label names
labels = ['label1','label2','label3']

## name of col with text in it
text_col = 'text_stop_stem'

## list of classifiers
classifiers = [ MultinomialNB(),
                LogisticRegression(max_iter = 5000),
                RandomForestClassifier(n_jobs = 2)]

## number of CV folds
n_folds = 10


# inits for storing data
df_metrics_list = [] # for model performance metrics
df_roc_pr_list = []  # for auc curves (ROC and PR)
df_coeffs_list = []  # cofficients of features

# loop over labels and classifiers
for i1, label in enumerate(labels):


    # loop over classifiers
    for i2, classifier in enumerate(classifiers):

        ## Run CV text classification
        temp = run_text_classifier_get_metrics(classifier, n_folds, df, text_col, label)
        
        ## save data
        df_metrics_list.append(temp[0])
        df_roc_pr_list.append(temp[1])
        df_coeffs_list.append(temp[2])
        
        ## print update
        print("Finished Classifier {} of {} for Label {} of {}"\
              .format(i2 + 1, len(classifiers), i1 + 1, len(labels)))
            
            
# unlist data
df_metrics = pd.concat(df_metrics_list)
df_roc_pr = pd.concat(df_roc_pr_list)
df_coeffs = pd.concat(df_coeffs_list)

# save as csv for analysis 
df_metrics.to_csv('../data/cv_metrics.csv')
df_roc_pr.to_csv('../data/cv_roc_pr.csv')
df_coeffs.to_csv('../data/cv_coeffs.csv')
