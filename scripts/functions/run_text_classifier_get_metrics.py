from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from statistics import harmonic_mean

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
from numpy.random import choice



# funcion to run CV text classification and get metrics

def run_text_classifier_get_metrics(classifier, # classifier function
                          n_folds, # numbe of CV folds
                          df, # data frame of data
                          text_col, # name of column with text data
                          label, # name of label 
                          decrease_n = False, # should we decrease training set sample size?
                          decrease_amnt = 0 # decrease training set sample size by how much?
                          ):


# --------------------------------------------------------------------------- #    
    # prep repsonse and explan variables
    
    ## get not nulls
    df_lab = df[df[label].notnull()]
    
    
    ## encode response
    Encoder = LabelEncoder()
    y = Encoder.fit_transform(df_lab[label])
    
    ## to sparse matrix for text data
    count_vect = CountVectorizer()
    count_vect.fit(df_lab[text_col])
    x_names = count_vect.get_feature_names()
    X = count_vect.transform(df_lab[text_col])
    
    
# --------------------------------------------------------------------------- #   

    # initialize variable to save data into
    
    ## metrics
    tp = [] # true positives
    fp = [] # false positives
    fn = [] # false negatives
    tn = [] # true negatives
    fold_a = [] # CV fold number
    auc_1 = [] # AUC from model
    avg_prec = [] # average precision from model
    

    ## ROC & PR curves
    tprs = [] # true positives
    fprs = [] # false positives
    fold_b = [] # CV fold number
    roc_aucs = [] # AUC values
    mean_fpr = np.linspace(0, 1, 100) # for AUC thresholds
    
    prs = [] # precision
    rcl = [] # recall
    pr_aucs = [] # AUC values
    mean_recall = np.linspace(0, 1, 100) # for AUC thresholds
    
    # feature coefficient values
    coeffs = [] # coefficient 
    feature_names = [] # feature names
    fold_c = [] # CV fold number
    
# --------------------------------------------------------------------------- #      
    
    # Run CV classification and save results for each fold
    
    ## set up CV split
    cv = StratifiedKFold(n_splits= n_folds )
    
    ## loop over CV fold
    for i, (train, test) in enumerate(cv.split(X, y)):
        
        # if running sensitivity analysis by decreasing sample set size
        if decrease_n:
            
            # randomly choose data points from training set
            train = choice(train, (train.shape[0] - decrease_amnt), replace = False)

        
        # run classifier ---------------------------------------------------- #
        classifier.fit(X[train], y[train])
        
        # get predictions --------------------------------------------------- #
        
        ## 1/0 prediction based on sci-kits predetermined cutoff (For confusion matrix)
        preds = classifier.predict(X[test])
        
        ## as a probability (For ROC and PR curves)
        probas_  = classifier.predict_proba(X[test])        
        
        # get feature coeffs ------------------------------------------------ #
        
        ## different method depending on classifier
        if classifier.__class__.__name__  in ["LogisticRegression", "MultinomialNB"]: 
            coeffs.append(classifier.coef_.tolist()[0])
            
        elif classifier.__class__.__name__ in ["RandomForestClassifier"] :
            coeffs.append(classifier.feature_importances_.tolist())
            
        ## save data    
        feature_names.append(x_names)
        fold_c.append([i+1] * len(x_names))
        
        
        # get metrics ------------------------------------------------------ #
        
        ## auc + avg prec
        auc_1.append(roc_auc_score(y[test], probas_[:, 1]))
        avg_prec.append( average_precision_score(y[test], probas_[:, 1]) )
        
        ## confusion matrix
        tn_temp, fp_temp, fn_temp, tp_temp = confusion_matrix(y[test], preds).ravel()
        # cm = confusion_matrix(y[test], preds)
        
        ## save data
        tp.append(tp_temp) # cm[0,0] 
        fn.append(fn_temp) # cm[1,0] 
        fp.append(fp_temp) # cm[0,1] 
        tn.append(tn_temp) # cm[1,1] 
        fold_a.append( i + 1 )
        
        # get roc & pr values ----------------------------------------------- #
        
        ## roc
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        fprs.append(mean_fpr) 
        roc_auc = auc(fpr, tpr)
        roc_aucs.append([roc_auc] * len(mean_fpr))

        ## prec recall
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
        pr_auc = auc(recall, precision)
        pr_aucs.append([pr_auc] * len(mean_recall))
        prs.append( np.interp(mean_recall, precision, recall) )
        rcl.append(mean_recall) 
        
        ## fold number
        fold_b.append([i+1] * len(mean_fpr))      
        
        
        # print update ------------------------------------------------------ #
        print('fold {} of {} done.'.format(i+1, n_folds))
        
        
        
# --------------------------------------------------------------------------- #           
    # Convert results to DataFrames
        
    ## metrics DataFrame ---------------------------------------------------- #
    df_metrics = pd.DataFrame({'label' : label,
                          'name' : classifier.__class__.__name__,
                          'decrease_amnt' : decrease_amnt,
                          'fold' : fold_a,
                          'auc' : auc_1,
                          'avg_prec' : avg_prec,
                          'tp' : tp,
                          'fn' : fn,
                          'fp' : fp,
                          'tn' : tn})
        

    ### calculate additional metrics
    df_metrics['tpr'] = df_metrics.tp / (df_metrics.tp + df_metrics.fn)
    df_metrics['fpr'] = df_metrics.fp / (df_metrics.fp + df_metrics.tn)
    df_metrics['prec'] = df_metrics.tp / (df_metrics.tp + df_metrics.fp)
    df_metrics['faor'] = df_metrics.fn / (df_metrics.fn + df_metrics.tn)
    df_metrics['f1'] = [ harmonic_mean([x,y]) for x, y in zip(df_metrics.tpr, df_metrics.prec) ]


    ## roc & pr DataFrame --------------------------------------------------- #
    df_roc_pr = pd.DataFrame({'label' : label,
                      'name' : classifier.__class__.__name__,
                      'decrease_amnt' : decrease_amnt,
                      'fold' : np.hstack( fold_b ), 
                      'roc_auc' : np.hstack( roc_aucs ),
                      'fpr' : np.hstack( fprs ),
                      'tpr' : np.hstack( tprs ),
                      'pr_auc' : np.hstack( pr_aucs ),
                      'precsion' : np.hstack( prs ),
                      'recall' : np.hstack( rcl )})

    # Feature coefficients DataFrame -----------------------------------------#
    coeffs_df = pd.DataFrame({'label' : label,
                             'name' : classifier.__class__.__name__,
                             'decrease_amnt' : decrease_amnt,
                             'fold' : np.hstack( fold_c ),
                             'feature' : np.hstack( feature_names ),
                             'coeff' : np.hstack( coeffs) })

# --------------------------------------------------------------------------- #   
    # return list of results
    return(df_metrics, df_roc_pr, coeffs_df)