# Text classification to streamline online wildlife trade analyses
Code and data for text classification models association with "Text classification to streamline online wildlife trade analyses". by Stringham OC, Moncayo S, Hill KGW, Toomes A, Mitchell L, Ross JV, Cassey P. (2021). PLOS ONE: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254007

---

## Content
The following contains descriptions of each file: 

- data/raw/text_w_labels.csv

    Data file containing the online listings described in the paper along with their associated text. Column metadata:

    1. classifieds_unique_listing_id: unique identifer for listings
    2. text_stop_stem: processed text, including remove special characters, numbers, stop words, and stemming.
    3. label1: The 'domestic poultry' label. 1 indicates the listing was labelled as domestic poultry. 
    4. label2: The 'junk; label. 1 indicates the listing was labelled as junk. 
    5. label3: the 'wanted' label. 1 indicates the listing was labelled as wanted.


- scripts/functions/run_text_classifier_get_metrics.py

    Function that performs the text classification and extracts: (1) model metrics, (2) data for ROC and PR curves, and (3) coefficient values for features. In addition, this function can perform the sensitivity mentioned in the paper. Metadata for function input parameters:

    - classifier: the function of the text classifier (e.g., MultinomialNB() from sklearn.naive_bayes)
    - n_folds: the number of cross-validated folds to perform
    - df: DataFrame of text data along with labels. Text must be in one column seperated by a space. Labels must be in their own column.
    - text_col: the name of column in `df` where the text data is stored
    - label: the name of the column in `df` where the label value is stored (as 0s and 1s)
    - decrease_n: Optional. True/False on whether or not to perform the sensitivity analysis (i.e., reduce the training set sample size).
    - decrease_amnt: Optional. How many rows to reducde the training set sample size by.

- scripts/cv_text_classification_all_labels_all_classifiers.py

    A script that performs the text classification for all labels ('domestic poultry', 'junk', and 'wanted' labels) and all classifiers (Logistic Regression, Naive Bayes, Random Forests). The script uses the `run_text_classifier_get_metrics` to run the text classification and generate the results.

- scripts/cv_text_classification_one_label_one_classifier.py

    A script that performs the text classification for one label and one classifier. 

- scripts/sensitivity_analysis.py

    A script that performs the sensitivity analysis mentioned in the paper.
