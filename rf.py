import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_curve


n_splits=50
test_size=0.20


def read_table(input_fn, sep='\t'):
    """Read table.
    """

    table = pd.read_csv(input_fn, sep=sep, index_col=0)
    table.index = [str(i) for i in table.index]
    
    return table


def read_problem(data_fn, labels_fn=None, target=None, sep='\t'):

    data_df = read_table(data_fn, sep=sep).T
       
    if labels_fn is not None:
        labels_df = read_table(labels_fn, sep=sep)

        if target is None:
            target = labels_df.columns[0]

        if target not in labels_df.columns:
            raise ValueError("target {} is not in the labels file".\
                             format(target))
        
        labels_s = labels_df.loc[labels_df[target].notnull(), target]
        sample_ids = sorted(set(data_df.index) & set(labels_s.index))
        data_df, labels_s = data_df.loc[sample_ids], labels_s[sample_ids]
    else:
        labels_s = None
        
    return data_df, labels_s


def rfclass(X_df, y_df, n_estimators=500, n_jobs=1, n_splits=10, test_size=0.1):

    X, y = X_df.values, y_df.values

    y = LabelEncoder().fit(y).transform(y)

    features = X_df.columns

    # set up the RF classifier, dummy classifiers and the cross-validator
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0,
        class_weight="balanced", n_jobs=n_jobs, oob_score=True)
    clf_bl = DummyClassifier(strategy='most_frequent', random_state=0)
    clf_rnd = DummyClassifier(strategy='stratified', random_state=0)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
        random_state=0)

    mean_fpr = np.linspace(0, 1, 100)
    metrics, roc_curves, importances = [], [], []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print (i)

        # train/test splits
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # RF classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        tpr = sp.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        oob_error = 1. - clf.oob_score_
  
        # 'most frequent' (baseline) dummy classifier
        clf_bl.fit(X_train, y_train)
        y_pred_bl = clf_bl.predict(X_test)
        y_proba_bl = clf_bl.predict_proba(X_test)
        acc_bl = accuracy_score(y_test, y_pred_bl)
        mcc_bl = matthews_corrcoef(y_test, y_pred_bl)
        fpr_bl, tpr_bl, _ = roc_curve(y_test, y_proba_bl[:, 1])
        tpr_bl = sp.interp(mean_fpr, fpr_bl, tpr_bl)
        tpr_bl[0] = 0.0
        
        # 'stratified' (random) dummy classifier
        clf_rnd.fit(X_train, y_train)
        y_pred_rnd = clf_rnd.predict(X_test)
        y_proba_rnd = clf_rnd.predict_proba(X_test)
        acc_rnd = accuracy_score(y_test, y_pred_rnd)
        mcc_rnd = matthews_corrcoef(y_test, y_pred_rnd)
        fpr_rnd, tpr_rnd, _ = roc_curve(y_test, y_proba_rnd[:, 1])
        tpr_rnd = sp.interp(mean_fpr, fpr_rnd, tpr_rnd)
        tpr_rnd[0] = 0.0

        # append results
        metrics.append([i, acc, acc_bl, acc_rnd, mcc, mcc_bl, mcc_rnd, oob_error])

        for j in range(len(mean_fpr)):
            roc_curves.append([i, mean_fpr[j], tpr[j], tpr_bl[j], tpr_rnd[j]])

        for k, fi in enumerate(clf.feature_importances_):
            importances.append([i, features[k], fi])

    metrics_df = pd.DataFrame(metrics, 
        columns=["CVIter", "Accuracy", "AccuracyBaseline", "AccuracyRandom",
                 "MCC", "MCCBaseline", "MCCRandom", "OOBError"])
    metrics_df.set_index(keys=["CVIter"], inplace=True)
    
    roc_curves_df = pd.DataFrame(roc_curves,
        columns=["CVIter", "FPR", "TPR", "TPRBaseline", "TPRRandom"])
    roc_curves_df.set_index(keys=["CVIter"], inplace=True)

    importances_df = pd.DataFrame(importances, 
        columns=["CVIter", "Feature", "Importance"])
    importances_df.set_index(keys=["CVIter", "Feature"],
        inplace=True)

    return metrics_df, roc_curves_df, importances_df


X, y = read_problem("16S_papio.tsv", labels_fn="metadata.txt",
    target="Forest_block")
metrics, roc_curves, importances = rfclass(X, y, n_estimators=500, n_jobs=1,
    n_splits=n_splits, test_size=test_size)
metrics.to_csv("metrics_papio_16S.txt", sep='\t', float_format="%.6f")
roc_curves.to_csv("roc_curves_papio_16S.txt", sep='\t', float_format="%.6f")
importances.to_csv("importances_papio_16S.txt", sep='\t', float_format="%.6f")

X, y = read_problem("16S_procolobus.tsv", labels_fn="metadata.txt", 
                    target="Forest_block")
metrics, roc_curves, importances = rfclass(X, y, n_estimators=500, n_jobs=1,
    n_splits=n_splits, test_size=test_size)
metrics.to_csv("metrics_procolobus_16S.txt", sep='\t', float_format="%.6f")
roc_curves.to_csv("roc_curves_procolobus_16S.txt", sep='\t', float_format="%.6f")
importances.to_csv("importances_procolobus_16S.txt", sep='\t', float_format="%.6f")

X, y = read_problem("ITS_papio.tsv", labels_fn="metadata.txt", 
                    target="Forest_block")
metrics, roc_curves, importances = rfclass(X, y, n_estimators=500, n_jobs=1,
    n_splits=n_splits, test_size=test_size)
metrics.to_csv("metrics_papio_ITS.txt", sep='\t', float_format="%.6f")
roc_curves.to_csv("roc_curves_papio_ITS.txt", sep='\t', float_format="%.6f")
importances.to_csv("importances_papio_ITS.txt", sep='\t', float_format="%.6f")

X, y = read_problem("ITS_procolobus.tsv", labels_fn="metadata.txt", 
                    target="Forest_block")
metrics, roc_curves, importances = rfclass(X, y, n_estimators=500, n_jobs=1,
    n_splits=n_splits, test_size=test_size)
metrics.to_csv("metrics_procolobus_ITS.txt", sep='\t', float_format="%.6f")
roc_curves.to_csv("roc_curves_procolobus_ITS.txt", sep='\t', float_format="%.6f")
importances.to_csv("importances_procolobus_ITS.txt", sep='\t', float_format="%.6f")
