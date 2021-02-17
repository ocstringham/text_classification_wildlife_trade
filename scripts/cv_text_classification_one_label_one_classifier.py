import pandas as pd

# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

from functions.run_text_classifier_get_metrics import run_text_classifier_get_metrics


# define inits
n_folds = 10
label = "label1"
classifier = LogisticRegression(max_iter = 5000) #RandomForestClassifier(n_jobs = 2) #MultinomialNB()

# load dataFrame of text data
file_loc = '../data/raw/text_w_labels.csv'
df = pd.read_csv( file_loc, encoding = "ISO-8859-1")
## name of col with text in it
text_col = 'text_stop_stem'

# run
temp = run_text_classifier_get_metrics(classifier, n_folds, df, text_col, label)

# extract metrics
df_metrics = temp[0]
df_roc_pr = temp[1]
df_coeffs = temp[2]
