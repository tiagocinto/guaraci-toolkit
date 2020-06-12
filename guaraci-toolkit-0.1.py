#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Guaraci Toolkit 0.1

Usage: python guaraci-toolkit-0.1.py -process mdl.sel -target flare_t1do

Targets:
- flare_t1do: t + 1d
- flare_t2do: t + 2d
- flare_t3do: t + 3d

Processes:
- out.kfd: Outer K-Fold
- out.kfd.ts: Outer K-Fold (Time Separated Periods)
- mdl.sel: Model Selection
- fs.flt: Univariate Filter Analysis
- fs.wrp: Wrapper Analysis
- hop.grd: Grid Analysis
- hop.val: Parameters Validation
- dat.rsp: Data Resampling
- cst.fun: Cost Function Analysis
- cfp.anl: Cut-Off Point Analysis
- cfp.val: Cut-Off Point Validation
- val.evl: Validation Sets Evaluation
- tst.evl: Test Sets Evaluation

This script automates the Guaraci framework when designing space weather predictors. It's optimized for Python 2.7 and unix-based operating systems.
To configure this script, got to the 'Script Input Data' section.

Author: Tiago Cinto
Version: 0.1
Email: tiago.cinto@pos.ft.unicamp.br
"""

from __future__ import division

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('[INFO] no display found... using non-interactive agg backend')
    mpl.use('Agg')

import argparse
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from operator import itemgetter
from multiprocessing import Process

from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import (SelectKBest, f_classif)
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, train_test_split)
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier)

from imblearn.combine import (SMOTEENN, SMOTETomek)
from imblearn.over_sampling import (RandomOverSampler, ADASYN, SMOTE)
from imblearn.under_sampling import (AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, OneSidedSelection, ClusterCentroids, RandomUnderSampler, NeighbourhoodCleaningRule, CondensedNearestNeighbour, NearMiss, InstanceHardnessThreshold)

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 14, 6


def append_text_to_file(file_name, text):
    """
    Appends some text content to a file. Used for outputting results.
    """
    f = open(file_name,'a')
    f.write(text)
    f.close()


def calc_tn(y_true, y_pred): 
    """
    Calculates the true negatives of a confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)[0, 0]


def calc_fp(y_true, y_pred): 
    """
    Calculates the false positives of a confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)[0, 1]


def calc_fn(y_true, y_pred): 
    """
    Calculates the false negatives of a confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)[1, 0]


def calc_tp(y_true, y_pred): 
    """
    Calculates the true positives of a confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)[1, 1]


def calc_cnf_matrix_elements(y_true, y_pred):
    """
    Calculates tn, fp, fn, and fp of a confusion matrix.
    """
    tn=calc_tn(y_true, y_pred)
    fp=calc_fp(y_true, y_pred)
    fn=calc_fn(y_true, y_pred)
    tp=calc_tp(y_true, y_pred)
    return tn, fp, fn, tp


def calc_scores_from_cnf_matrix(cnf_matrix):  
    """
    Calculates the scores based on a confusion matrix.
    """
    tn, fp, fn, tp = cnf_matrix.ravel() 
    acc = safe_division((tp+tn),(tp+tn+fp+fn))
    tpr = safe_division(tp,(tp+fn))
    tnr = safe_division(tn,(tn+fp))
    ppv = safe_division(tp,(tp+fp))
    npv = safe_division(tn,(tn+fn))
    far = safe_division(fp,(tp+fp))
    f1 = safe_division(2*tp,(2*tp+fn+fp))
    tss = tpr + tnr - 1	
    hss = safe_division(2*((tp*tn)-(fn*fp)),((tp+fn)*(fn+tn))+((tp+fp)*(fp+tn)))
    return acc, tpr, tnr, ppv, npv, far, tss, f1, hss
    

def calc_scores_from_cnf_matrix_elements(tn, fp, fn, tp):  
    """
    Calculates the scores based on the confusion matrix elements.
    """
    acc = safe_division((tp+tn),(tp+tn+fp+fn))
    tpr = safe_division(tp,(tp+fn))
    tnr = safe_division(tn,(tn+fp))
    ppv = safe_division(tp,(tp+fp))
    npv = safe_division(tn,(tn+fn))
    far = safe_division(fp,(tp+fp))
    f1 = safe_division(2*tp,(2*tp+fn+fp))
    tss = tpr + tnr - 1	
    hss = safe_division(2*((tp*tn)-(fn*fp)),((tp+fn)*(fn+tn))+((tp+fp)*(fp+tn)))
    return acc, tpr, tnr, ppv, npv, far, tss, f1, hss


def fit_model(alg, x_train, x_heldout, y_train, y_heldout, t=None):    
    """
    Fits a learning algorithm and forecasts a set of data. Set t for thresholding predictions.
    """
    alg.fit(x_train, y_train)
    y_predprob = alg.predict_proba(x_heldout)
    if t == None:
        y_pred = alg.predict(x_heldout)
        cnf_matrix = confusion_matrix(y_heldout, y_pred)
    else:
        y_pred_thres = threshold_predict(y_scores=y_predprob[:,1], t=t)
        cnf_matrix = confusion_matrix(y_heldout, y_pred_thres)
    auc = roc_auc_score(y_heldout, y_predprob[:,1])
    ap = average_precision_score(y_heldout, y_predprob[:,1])
    acc, tpr, tnr, ppv, npv, far, tss, f1, hss = calc_scores_from_cnf_matrix(cnf_matrix)       
    return acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap
  
    
def get_model(cls):
    """
    Factory of baseline models.
    """
    if cls=='baseline-gbm':
         gbm = GradientBoostingClassifier(n_estimators=5,
                                          learning_rate=2.0,
                                          min_samples_split=7,
                                          min_samples_leaf=5,
                                          max_depth=3,
                                          max_features=2,
                                          subsample=0.5,
                                          random_state=10)
         return gbm
    elif cls=='baseline-ada':
       ada = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=7,
                                                       min_samples_leaf=5,
                                                       max_depth=3,
                                                       max_features=2, 
                                                       random_state=10),
                                n_estimators=5,
                                learning_rate=2.0,
                                random_state=10)
       return ada
    elif cls=='baseline-forest':
        forest = RandomForestClassifier(n_estimators=5,
                                        min_samples_split=7,
                                        min_samples_leaf=5,
                                        max_depth=3,
                                        max_features=2,
                                        min_impurity_split=0.25, 
                                        random_state=10)
        return forest
    
    
def grid_search_cv(data_x, data_y, param_grid, model, n_folds, scoring, n_scores, verbose):
    """
    Performs a grid search for hyperparameters of a given model based on a provided param grid.
    """
    grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           scoring=scoring, 
                           n_jobs=4,
                           iid=False, 
                           cv=n_folds,
                           verbose=2)
    grid_search.fit(data_x,data_y)
    print_grid_search_cv_report(grid_search.grid_scores_, n_top=n_scores)  
    
    
def heldout_split(data_x, data_y, test_size):
    """
    Performs an unique heldout split.
    """
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, stratify=data_y, shuffle=True)
    return x_train, x_test, y_train, y_test
    

def nested_grid_search_cv(data_x, data_y, param_grid, model, n_folds, scoring):
    """
    Performs a nested grid search for hyperparameters of a given model based on a provided param grid.
    """
    outer_scores=[]
    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=10)
    
    #outer folds
    for i, (train_idx, heldout_idx) in enumerate(outer_cv.split(data_x, data_y)):
        print('[Outer fold %d/5]' % (i+1))
        x_train, x_heldout = data_x.iloc[train_idx], data_x.iloc[heldout_idx]
        y_train, y_heldout = data_y[train_idx], data_y[heldout_idx]    
    
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=n_folds, n_jobs=1, verbose=0)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        best_clf.fit(x_train, y_train)
        outer_scores.append(best_clf.score(x_heldout, y_heldout))
        print('CV score: %.2f (params=%s)' % (outer_scores[i], grid_search.best_params_))
        print('\nCV score: %.2f (5x5 nested CV)' % np.mean(outer_scores))    


def outer_data_split_time_separated(header, save_path, df_data_train, df_data_test, kf_splits, predictors, target, test_size):
    """
    Splits data into training, validation, and testing segments (time separated). Saves data segments to files.
    """
    data_x = df_data_train[predictors]
    data_y = df_data_train[target]
    folds = StratifiedKFold(n_splits=kf_splits, shuffle=True, random_state=10)
    i = 0
    for train_index, heldout_index in folds.split(data_x, data_y):
        i = i + 1
        x_train, x_heldout = data_x.iloc[train_index], data_x.iloc[heldout_index]
        y_train, y_heldout = data_y[train_index], data_y[heldout_index]
        np.savetxt(save_path + 'data_train.fold.' + str(i) +'.csv', x_train.join(y_train), delimiter=';', fmt='%6f', comments='', header=header)
        np.savetxt(save_path + 'data_val.fold.' + str(i) +'.csv', x_heldout.join(y_heldout), delimiter=';', fmt='%6f', comments='', header=header)
        
    shuffled_df = df_data_test.sample(frac=1)
    test_folds = np.array_split(shuffled_df, 5)  
    i = 0
    for fold in test_folds:
        i = i + 1
        np.savetxt(save_path + 'data_test.fold.' + str(i) +'.csv', fold, delimiter=';', fmt='%6f', comments='', header=header)
    
    
def outer_data_split(header, save_path, data_x, data_y, kf_splits, test_size, predictors, target):
    """
    Splits data into training, validation, and testing segments. Saves data segments to files.
    """
    for i in range (1, kf_splits+1):    
        x_train, x_test, y_train, y_test = heldout_split(data_x=data_x, data_y=data_y, test_size=test_size)
        np.savetxt(save_path + 'data_train_full.fold.' + str(i) +'.csv', x_train.join(y_train), delimiter=';', fmt='%6f', comments='', header=header)
        np.savetxt(save_path + 'data_test.fold.' + str(i) +'.csv', x_test.join(y_test), delimiter=';', fmt='%6f', comments='', header=header)
        data_x = x_train
        data_y = y_train 
        test_size = test_size + 0.0025
    
    df_train = pd.read_csv(save_path + 'data_train_full.fold.' + str(kf_splits) + '.csv', sep=';', decimal='.')
    data_x = df_train[predictors]
    data_y = df_train[target]
    folds = StratifiedKFold(n_splits=kf_splits, shuffle=True, random_state=10)
    i = 0
    for train_index, heldout_index in folds.split(data_x, data_y):
        i = i + 1
        x_train, x_heldout = data_x.iloc[train_index], data_x.iloc[heldout_index]
        y_train, y_heldout = data_y[train_index], data_y[heldout_index]
        np.savetxt(save_path + 'data_train.fold.' + str(i) +'.csv', x_train.join(y_train), delimiter=';', fmt='%6f', comments='', header=header)
        np.savetxt(save_path + 'data_val.fold.' + str(i) +'.csv', x_heldout.join(y_heldout), delimiter=';', fmt='%6f', comments='', header=header)

     
class process_parameters_validation(Process):
    """
    Multi-job process: k-fold cross-validation of fine-tunned hyperparameters.
    """
    def __init__(self, model, df_train, n_splits, n_iterations, n_fold, log_path, predictors_subset, target, lock):
        super(process_parameters_validation, self).__init__()
        self.model = model
        self.df_train = df_train
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.predictors_subset = predictors_subset
        self.target = target
        self.lock = lock
    def run(self):
        print("\nworking on n_fold=" + str(self.n_fold) + "...")
        acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(alg=self.model, 
                                                                                                         data_x=self.df_train[self.predictors_subset], 
                                                                                                         data_y=self.df_train[self.target], 
                                                                                                         n_splits=5, 
                                                                                                         n_iterations=self.n_iterations, 
                                                                                                         print_inner_results=False)
        self.lock.acquire()
        log_entry = str('%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (self.n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
        append_text_to_file(self.log_path, log_entry + '\n')
        self.lock.release()
        print("\nprocess n_fold=" + str(self.n_fold) + " done!")


class process_univariate_feature_analysis(Process):
    """
    Multi-job process: performs an univariate feature analysis over the original set of features. Returns the set of features linked the highest TSS.
    """
    def __init__(self, model, data_x, data_y, n_splits, n_iterations, n_fold, log_path, max_features, lock):
        super(process_univariate_feature_analysis, self).__init__()
        self.model = model
        self.data_x = data_x
        self.data_y = data_y
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.max_features = max_features
        self.lock = lock
    def run(self):  
        r, c = self.data_x.shape
        i_ref = 0
        tss_ref = 0    
        tss_series = []
        hss_series = []
        i_series = []
    
        for i in range(self.max_features, c+1, 1):
            print('working on k: ' + str(i) + ' | fold: ' + str(self.n_fold))
            df_kbest = SelectKBest(f_classif, k=i).fit_transform(X=self.data_x, y=self.data_y)
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(alg=self.model, 
                                                                                                             data_x=pd.DataFrame(df_kbest), 
                                                                                                             data_y=self.data_y, 
                                                                                                             n_splits=self.n_splits, 
                                                                                                             n_iterations=self.n_iterations, 
                                                                                                             print_inner_results=False)
            i_series = np.append(i_series, i)
            tss_series = np.append(tss_series, tss)
            hss_series = np.append(hss_series, hss)
            if tss > tss_ref:
                tss_ref = tss
                i_ref = i
        
        self.lock.acquire()
        append_text_to_file(log_path, '\n...::: best K: %i | TSS: %.4f | n_fold: %i :::... \n' % (i_ref, tss_ref, self.n_fold))
        append_text_to_file(log_path, 'selected features for K: %d \n' % (i_ref))
                
        df_kbest = SelectKBest(f_classif, k=i_ref).fit(X=self.data_x, y=self.data_y)
        df_kbest_support = df_kbest.get_support()
        df_kbest_selected_features = [f for i, f in enumerate(self.data_x.columns) if df_kbest_support[i]]
        
        for n, s in zip(df_kbest_selected_features, df_kbest.scores_):
            append_text_to_file(log_path, '%s : %3.2f \n' % (n, s))
        self.lock.release()
   
     
class process_wrapper_analysis(Process):
    """
    Multi-job process: k-fold cross-validation with the subset of features.
    """
    def __init__(self, model, df_train, predictors_subset, target, n_splits, n_iterations, n_fold, log_path, lock):
        super(process_wrapper_analysis, self).__init__()
        self.model = model
        self.df_train = df_train
        self.predictors_subset = predictors_subset
        self.target = target
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.lock = lock
    def run(self):  
        print("\nworking on n_fold=" + str(self.n_fold) + "...")
        acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(alg=self.model, 
                                                                                                         data_x=self.df_train[self.predictors_subset], 
                                                                                                         data_y=self.df_train[self.target], 
                                                                                                         n_splits=5, 
                                                                                                         n_iterations=self.n_iterations, 
                                                                                                         print_inner_results=False)
        self.lock.acquire()
        log_entry = str('%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (self.n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
        append_text_to_file(log_path, log_entry + '\n')
        self.lock.release()
        print("\nprocess n_fold=" + str(self.n_fold) + " done!")


class process_spot_check_models(Process):
    """
    Multi-job process: performs a model selection with distinct models. Returns the model with the highest TSS.
    """
    def __init__(self, models, df_train, predictors, target, n_splits, n_iterations, n_fold, log_path, lock):
        super(process_spot_check_models, self).__init__()
        self.models = models
        self.df_train = df_train
        self.predictors = predictors
        self.target = target
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.lock = lock
    def run(self):  
        for clf in self.models:
            model = get_model(clf)
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, log = repeated_stratified_k_fold_validation(alg=model, 
                                                                                                             data_x=self.df_train[self.predictors],
                                                                                                             data_y=self.df_train[self.target], 
                                                                                                             n_splits=self.n_splits, 
                                                                                                             n_iterations=self.n_iterations, 
                                                                                                             print_inner_results=False)
            self.lock.acquire()
            log_entry = str('%s;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (clf,self.n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
            append_text_to_file(log_path, log_entry + '\n')
            self.lock.release()
            print("\nn_fold=" + str(self.n_fold) + " | model=" + clf + " done!")
           
            
class process_cost_function_validation(Process):
    """
    Multi-job process: k-fold cross-validation with the custom cp/cn ratio.
    """
    def __init__(self, model, df_train, n_splits, n_iterations, n_fold, log_path, predictors, target, c_ratio, cost_function_param_name, imbalanced_positive, lock):
        super(process_cost_function_validation, self).__init__()
        self.model = model
        self.df_train = df_train
        self.predictors = predictors
        self.target = target
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.c_ratio = c_ratio
        self.cost_function_param_name = cost_function_param_name
        self.imbalanced_positive = imbalanced_positive
        self.lock = lock
    def run(self):    
        print("\nworking on n_fold=" + str(self.n_fold) + "...")
        if self.imbalanced_positive:
            custom_class_ratio = {self.cost_function_param_name:{1:1,0:'%.2f' % (self.c_ratio)}}
        else:
            custom_class_ratio = {self.cost_function_param_name:{0:1,1:'%.2f' % (self.c_ratio)}}
        self.model.set_params(**custom_class_ratio)
        acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(alg=self.model, 
                                                                                                         data_x=self.df_train[self.predictors], 
                                                                                                         data_y=self.df_train[self.target],
                                                                                                         n_splits=5, 
                                                                                                         n_iterations=self.n_iterations, 
                                                                                                         print_inner_results=False)
        self.lock.acquire()
        log_entry = str('%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (self.n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
        append_text_to_file(log_path, log_entry + '\n')    
        self.lock.release()
        print("\nprocess n_fold=" + str(self.n_fold) + " done!")
        
        
class process_cut_off_point_validation(Process):
    """
    Multi-job process: k-fold cross-validation with the custom prediction threshold.
    """
    def __init__(self, t, model, df_train, n_splits, n_iterations, n_fold, log_path, predictors, target, cost_function_param_name, imbalanced_positive, c_ratio, resampling_method, lock):
        super(process_cut_off_point_validation, self).__init__()
        self.t = t
        self.model = model
        self.df_train = df_train
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.predictors = predictors
        self.target = target
        self.cost_function_param_name = cost_function_param_name
        self.imbalanced_positive = imbalanced_positive 
        self.c_ratio = c_ratio
        self.resampling_method = resampling_method
        self.lock = lock
    def run(self):     
        print("\nworking on n_fold=" + str(self.n_fold) + "...")
        if self.c_ratio != None:
            if self.imbalanced_positive:
                custom_class_ratio = {self.cost_function_param_name:{1:1,0:'%.2f' % (self.c_ratio)}}
            else:
                custom_class_ratio = {self.cost_function_param_name:{0:1,1:'%.2f' % (self.c_ratio)}}
            self.model.set_params(**custom_class_ratio)
        
        if self.resampling_method != None:
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = resampled_repeated_stratified_k_fold_validation(method=self.resampling_method,
                                                                                                                       n_iterations=self.n_iterations, 
                                                                                                                       alg=self.model, 
                                                                                                                       data_x=self.df_train[self.predictors], 
                                                                                                                       data_y=self.df_train[self.target], 
                                                                                                                       n_splits=self.n_splits, 
                                                                                                                       t=self.t, 
                                                                                                                       print_inner_results=False)
        else:
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(n_iterations=self.n_iterations, 
                                                                                                             alg=self.model, 
                                                                                                             data_x=self.df_train[self.predictors], 
                                                                                                             data_y=self.df_train[self.target], 
                                                                                                             n_splits=self.n_splits, 
                                                                                                             t=self.t, 
                                                                                                             print_inner_results=False)
        
        self.lock.acquire()
        log_entry = str('%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (self.n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
        append_text_to_file(log_path, log_entry + '\n')   
        self.lock.release()
        print("\nprocess n_fold=" + str(self.n_fold) + " done!")
        
        
class process_data_resampling(Process):
    """
    Multi-job process: performs repeated k-fold cross-validation with data resampling.
    """
    def __init__(self, model, methods, df_train, predictors, target, n_splits, n_iterations, n_fold, log_path, lock):
        super(process_data_resampling, self).__init__()
        self.model = model
        self.methods = methods
        self.df_train = df_train
        self.predictors = predictors
        self.target = target
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.lock = lock
    def run(self): 
        for i, m in enumerate(self.methods):
            print("\nworking on n_fold=" + str(self.n_fold) + " | method: " + m + "...")
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = resampled_repeated_stratified_k_fold_validation(method=m, 
                                                                                                                       alg=self.model, 
                                                                                                                       data_x=self.df_train[self.predictors], 
                                                                                                                       data_y=self.df_train[self.target], 
                                                                                                                       n_splits=5, 
                                                                                                                       n_iterations=self.n_iterations, 
                                                                                                                       print_inner_results=False)
            self.lock.acquire()                                                                                                    
            log_entry = str('%s;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (m,self.n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
            append_text_to_file(log_path, log_entry + '\n')
            self.lock.release()
            print("\nprocess n_fold=" + str(self.n_fold) + " | method: " + m + " done!")
    
    
class process_cost_function_analysis(Process):
    """
    Multi-job process: performs a cost function analysis seeking the cp/cn ratio linked to an intersection TPR x TNR. 
    """
    def __init__(self, model, df_train, predictors, target, n_splits, n_iterations, n_fold, log_path, r_start, r_end, r_step, cost_function_param_name, lock, imbalanced_positive=True):
        super(process_cost_function_analysis, self).__init__()
        self.model = model
        self.df_train = df_train
        self.predictors = predictors
        self.target = target
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.log_path = log_path
        self.r_start = r_start
        self.r_end = r_end
        self.r_step = r_step
        self.cost_function_param_name = cost_function_param_name
        self.imbalanced_positive = imbalanced_positive
        self.lock = lock
    def run(self):       
        i_ref = 0
        dif_ref =  100.0
        tss_series = []
        tnr_series = []
        tpr_series = []
        i_series = []
        for i in np.arange(self.r_start, self.r_end + self.r_step, self.r_step):
            if self.imbalanced_positive:
                custom_class_ratio = {self.cost_function_param_name:{1:1,0:'%.2f' % (i)}}
            else:
                custom_class_ratio = {self.cost_function_param_name:{0:1,1:'%.2f' % (i)}}
            self.model.set_params(**custom_class_ratio)
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(alg=self.model, 
                                                                                                             data_x=self.df_train[self.predictors], 
                                                                                                             data_y=self.df_train[self.target], 
                                                                                                             n_splits=self.n_splits, 
                                                                                                             n_iterations=self.n_iterations, 
                                                                                                             print_inner_results=False)
            i_series = np.append(i_series, i)
            tss_series = np.append(tss_series, tss)
            tnr_series = np.append(tnr_series, tnr)
            tpr_series = np.append(tpr_series, tpr)
            dif = abs(tpr - tnr)
            print('\nn_fold: %d | cp/cn ratio: %.2f | tss: %.2f | tpr: %.2f | tnr: %.2f ' % (self.n_fold, i, tss, tpr, tnr))
            if dif != 0 and dif < dif_ref:
                dif_ref = dif
                i_ref = i      
        if self.imbalanced_positive:
            custom_class_ratio = {self.cost_function_param_name:{1:1,0:'%.2f' % (i)}}
        else:
            custom_class_ratio = {self.cost_function_param_name:{0:1,1:'%.2f' % (i)}}
        self.model.set_params(**custom_class_ratio)
        acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, log = repeated_stratified_k_fold_validation(alg=self.model, 
                                                                                                         data_x=self.df_train[self.predictors], 
                                                                                                         data_y=self.df_train[self.target], 
                                                                                                         n_splits=self.n_splits, 
                                                                                                         n_iterations=self.n_iterations, 
                                                                                                         print_inner_results=False)
        self.lock.acquire()
        append_text_to_file(log_path, "\nc_ref: %.2f | n_fold: %d | tss: %.2f | tpr: %.2f | tnr: %.2f" % (i_ref, self.n_fold, tss, tpr, tnr))
        append_text_to_file(log_path, log)
        self.lock.release()
        
        plt.figure()
        plt.plot(i_series, tss_series, linewidth=3.0, color='blue', label='TSS')
        plt.legend()
        plt.plot(i_series, tnr_series, linewidth=3.0, color='red', label='TNR')
        plt.legend()
        plt.plot(i_series, tpr_series, linewidth=3.0, color='green', label='TPR')
        plt.legend()
        plt.xlabel("Class ratio Cp/Cn")
        plt.ylabel("Skill score")
        plt.loc='best'
        plt.grid(True)
        plt.xlim(self.r_start,self.r_end)
        plt.savefig(log_path + '.graph.n_fold.%s.eps' % (self.n_fold), format='eps', dpi=1200)  


class process_cut_off_point_analysis(Process):
    """
    Multi-job process: performs a cut off point analysis seeking the threshold linked to an intersection TPR x PPV. 
    """   
    def __init__(self, model, df_train, predictors, target, n_splits, n_iterations, n_fold, t_start, t_end, t_step, log_path, cost_function_param_name, imbalanced_positive, c_ratio, resampling_method, lock):
        super(process_cut_off_point_analysis, self).__init__()
        self.model = model
        self.df_train = df_train
        self.predictors = predictors
        self.target = target
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.n_fold = n_fold
        self.t_start = t_start
        self.t_end = t_end
        self.t_step = t_step
        self.log_path = log_path
        self.cost_function_param_name = cost_function_param_name
        self.imbalanced_positive = imbalanced_positive
        self.resampling_method = resampling_method
        self.c_ratio = c_ratio
        self.lock = lock
    def run(self):
        i_ref = 0
        tpr_ref = 0
        ppv_ref = 0
        tss_ref = 0
        dif_ref =  100.0
        tss_series = []
        ppv_series = []
        tpr_series = []
        i_series = []
    
        if self.c_ratio != None:
            if self.imbalanced_positive:
                custom_class_ratio = {self.cost_function_param_name:{1:1,0:'%.2f' % (self.c_ratio)}}
            else:
                custom_class_ratio = {self.cost_function_param_name:{0:1,1:'%.2f' % (self.c_ratio)}}
            self.model.set_params(**custom_class_ratio)
        
        for i in np.arange(self.t_start, self.t_end + self.t_step, self.t_step):
            if self.resampling_method != None:
                acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = resampled_repeated_stratified_k_fold_validation(method=self.resampling_method,
                                                                                                                           n_iterations=self.n_iterations, 
                                                                                                                           alg=self.model, 
                                                                                                                           data_x=self.df_train[self.predictors], 
                                                                                                                           data_y=self.df_train[self.target], 
                                                                                                                           n_splits=self.n_splits, 
                                                                                                                           t=i, 
                                                                                                                           print_inner_results=False)
            else:
                acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, out = repeated_stratified_k_fold_validation(n_iterations=self.n_iterations, 
                                                                                                                 alg=self.model, 
                                                                                                                 data_x=self.df_train[self.predictors], 
                                                                                                                 data_y=self.df_train[self.target], 
                                                                                                                 n_splits=self.n_splits, 
                                                                                                                 t=i, 
                                                                                                                 print_inner_results=False)
            i_series = np.append(i_series, i)
            tss_series = np.append(tss_series, tss)
            ppv_series = np.append(ppv_series, ppv)
            tpr_series = np.append(tpr_series, tpr)
            print('\nn_fold: %d | t: %.2f | tss: %.2f | tpr: %.2f | ppv: %.2f ' % (self.n_fold, i, tss, tpr, ppv))
            dif = abs(tpr - ppv)
            
            if dif != 0 and tpr != 0 and ppv != 0:
                if dif < dif_ref:
                    if tss > tss_ref:
                        dif_ref = dif
                        i_ref = i
                        tss_ref = tss
                        tpr_ref = tpr
                        ppv_ref = ppv

                        
        self.lock.acquire()
        append_text_to_file(log_path, "\nt_ref: %.2f | n_fold: %d | tss: %.2f | tpr: %.2f | ppv: %.2f" % (i_ref, self.n_fold, tss_ref, tpr_ref, ppv_ref))
        append_text_to_file(log_path, out)
        self.lock.release()
        
        plt.figure()
        plt.plot(i_series, tss_series, linewidth=3.0, color='blue', label='TSS')
        plt.legend()
        plt.plot(i_series, ppv_series, linewidth=3.0, color='red', label='PPV')
        plt.legend()
        plt.plot(i_series, tpr_series, linewidth=3.0, color='green', label='TPR')
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Skill score")
        plt.loc='best'
        plt.grid(True)
        plt.xlim(self.t_start,self.t_end)
        plt.savefig(log_path + '.graph.n_fold.%s.eps' % (self.n_fold), format='eps', dpi=1200)  


def print_scores_report_basic(acc, tpr, tnr, ppv, npv, far, tss, f1, hss, title=None):
    """
    Print scores report of deterministic metrics only.
    """
    print(title)
    print("ACC: %.4f" % (acc))
    print("TPR: %.4f" % (tpr))
    print("TNR: %.4f" % (tnr))
    print("PPV: %.4f" % (ppv))
    print("NPV: %.4f" % (npv))
    print("FAR: %.4f" % (far))
    print("TSS: %.4f" % (tss))
    print("F1: %.4f" % (f1))
    print("HSS: %.4f" % (hss))
    
    
def print_scores_report(acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap, title=None):
    """
    Print scores report of both deterministic and probabilistic metrics.
    """
    print(title)
    print("ACC: %.4f" % (acc))
    print("TPR: %.4f" % (tpr))
    print("TNR: %.4f" % (tnr))
    print("PPV: %.4f" % (ppv))
    print("NPV: %.4f" % (npv))
    print("FAR: %.4f" % (far))
    print("TSS: %.4f" % (tss))
    print("F1: %.4f" % (f1))
    print("HSS: %.4f" % (hss))
    print("AUC: %.4f" % (auc))
    print("AP: %.4f" % (ap))       
    
    
def print_grid_search_cv_report(grid_scores, n_top=3, log_path=None):
    """
    Prints grid search cv report 
    """
    params = None
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        
        if log_path != None:
            append_text_to_file(log_path, "Parameters with rank: {0}\n".format(i + 1))
            append_text_to_file(log_path, "Mean validation score: {0:.4f} (std: {1:.4f})\n".format(score.mean_validation_score, np.std(score.cv_validation_scores)))
            append_text_to_file(log_path, "Parameters: {0}\n".format(score.parameters))
            
        if params == None:
            params = score.parameters
    
    return params


def print_cross_validation_scores(results):
    """
    Prints a report for cross-validation.
    """
    output=str("ACC: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,0]), np.std(results[:,0]), min(results[:,0]), max(results[:,0])))
    output+="\n"+str("TPR: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,1]), np.std(results[:,1]), min(results[:,1]), max(results[:,1])))
    output+="\n"+str("TNR: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,2]), np.std(results[:,2]), min(results[:,2]), max(results[:,2])))
    output+="\n"+str("PPV: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,3]), np.std(results[:,3]), min(results[:,3]), max(results[:,3])))
    output+="\n"+str("NPV: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,4]), np.std(results[:,4]), min(results[:,4]), max(results[:,4])))
    output+="\n"+str("FAR: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,5]), np.std(results[:,5]), min(results[:,5]), max(results[:,5])))
    output+="\n"+str("TSS: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,6]), np.std(results[:,6]), min(results[:,6]), max(results[:,6])))
    output+="\n"+str("F1: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,7]), np.std(results[:,7]), min(results[:,7]), max(results[:,7])))
    output+="\n"+str("HSS: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,8]), np.std(results[:,8]), min(results[:,8]), max(results[:,8])))
    output+="\n"+str("AUC: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,9]), np.std(results[:,9]), min(results[:,9]), max(results[:,9])))
    output+="\n"+str("AP: %.4f +/- %.4f [%.4f - %.4f]" % (np.mean(results[:,10]), np.std(results[:,10]), min(results[:,10]), max(results[:,10])))
    print(str(output[0:]))
    return str(output[0:])

        
def randomized_search_cv(data_x, data_y, param_grid, model, n_iter, n_folds, scoring, n_scores, verbose, log_path):
    """
    Performs a randomized search for hyperparameters of a given model based on a provided param grid.
    """
    random_search = RandomizedSearchCV(estimator=model, 
                                       param_distributions=param_grid, 
                                       scoring=scoring, 
                                       n_iter=n_iter,
                                       n_jobs=8,
                                       iid=False, 
                                       cv=n_folds,
                                       verbose=2)
    random_search.fit(data_x,data_y)
    print_grid_search_cv_report(random_search.grid_scores_, n_top=n_scores, log_path=log_path)   
   
    
def repeated_stratified_k_fold_validation(alg, data_x, data_y, n_splits, n_iterations, t=None, print_inner_results=False):
    """
    Performs repeated stratified k-fold validation of a model over a set of data. Set t for thresholding predictions.
    """
    log=""
    inner_results = np.empty(shape=[0, 11])
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_iterations, random_state=10)
    
    for train_index, heldout_index in folds.split(data_x, data_y):
        x_train, x_heldout = data_x.iloc[train_index], data_x.iloc[heldout_index]
        y_train, y_heldout = data_y.iloc[train_index], data_y.iloc[heldout_index]
        
        if t == None:
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg, x_train, x_heldout, y_train, y_heldout)
        else: 
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg, x_train, x_heldout, y_train, y_heldout, t=t)
        
        inner_results = np.vstack([inner_results, [acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap]])
        
    if print_inner_results:
        log = print_cross_validation_scores(inner_results)
    
    return np.mean(inner_results[:,0]), np.mean(inner_results[:,1]), np.mean(inner_results[:,2]), np.mean(inner_results[:,3]), np.mean(inner_results[:,4]), np.mean(inner_results[:,5]), np.mean(inner_results[:,6]), np.mean(inner_results[:,7]), np.mean(inner_results[:,8]), np.mean(inner_results[:,9]), np.mean(inner_results[:,10]), log


def resampled_repeated_stratified_k_fold_validation(method, alg, data_x, data_y, n_splits, n_iterations, print_inner_results=False, t=None):
    """
    Performs resampled repeated stratified k-fold validation of a model over a set of data. Training data are first resampled and then model is fit. Set t for thresholding predictions.
    """
    log=""
    inner_results = np.empty(shape=[0, 11])
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_iterations, random_state=10)
    
    for train_index, heldout_index in folds.split(data_x, data_y):
        x_train, x_heldout = data_x.iloc[train_index], data_x.iloc[heldout_index]
        y_train, y_heldout = data_y.iloc[train_index], data_y.iloc[heldout_index]
        
        x_resampled, y_resampled = resample_data(data_x=x_train, data_y=y_train, method=method)    
        
        if t == None:
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg, x_resampled, x_heldout, y_resampled, y_heldout)
        else:
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg, x_resampled, x_heldout, y_resampled, y_heldout, t=t)
        
        inner_results = np.vstack([inner_results, [acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap]])
        
    if print_inner_results:
        log = print_cross_validation_scores(inner_results)
    
    return np.mean(inner_results[:,0]), np.mean(inner_results[:,1]), np.mean(inner_results[:,2]), np.mean(inner_results[:,3]), np.mean(inner_results[:,4]), np.mean(inner_results[:,5]), np.mean(inner_results[:,6]), np.mean(inner_results[:,7]), np.mean(inner_results[:,8]), np.mean(inner_results[:,9]), np.mean(inner_results[:,10]), log


def resample_data(data_x, data_y, method):
    """
    Resamples a set of data based on a provided resampling method.
    """
    if method=='adasyn':
        util = ADASYN()
    elif method=='random-over-sampler':
        util = RandomOverSampler()
    elif method=='smote':
        util = SMOTE(kind='borderline2')
    elif method=='smote-tomek':
        util = SMOTETomek()
    elif method=='smote-enn':
        util = SMOTEENN()
    elif method=='edited-nn':
        util = EditedNearestNeighbours()
    elif method=='repeated-edited-nn':
        util = RepeatedEditedNearestNeighbours()
    elif method=='all-knn':
        util = AllKNN()
    elif method=='one-sided-selection':
        util = OneSidedSelection()
    elif method=='cluster-centroids':
        util = ClusterCentroids()
    elif method=='random-under-sampler':
        util = RandomUnderSampler()
    elif method=='neighbourhood-cleaning-rule':
        util = NeighbourhoodCleaningRule()
    elif method=='condensed-nearest-neighbour':
        util = CondensedNearestNeighbour()
    elif method=='near-miss':
        util = NearMiss(version=1)
    elif method=='instance-hardness-threshold':
        util = InstanceHardnessThreshold()
    
    x_resampled, y_resampled = util.fit_sample(data_x, data_y)
    
    return pd.DataFrame(x_resampled), pd.DataFrame(y_resampled)


def stratified_k_fold_validation(alg, data_x, data_y, n_splits, t=None, print_inner_results=False):
    """
    Performs a stratified k-fold validation of a model over a set of data. Set t for thresholding predictions.
    """
    log=""
    inner_results = np.empty(shape=[0, 11])
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    
    for train_index, heldout_index in folds.split(data_x, data_y):
        x_train, x_heldout = data_x.iloc[train_index], data_x.iloc[heldout_index]
        y_train, y_heldout = data_y.iloc[train_index], data_y.iloc[heldout_index]
        
        if t == None:
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg, x_train, x_heldout, y_train, y_heldout)
        else: 
            acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg, x_train, x_heldout, y_train, y_heldout, t=t)
        inner_results = np.vstack([inner_results, [acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap]])
        
    if print_inner_results:
        log = print_cross_validation_scores(inner_results)
    
    return np.mean(inner_results[:,0]), np.mean(inner_results[:,1]), np.mean(inner_results[:,2]), np.mean(inner_results[:,3]), np.mean(inner_results[:,4]), np.mean(inner_results[:,5]), np.mean(inner_results[:,6]), np.mean(inner_results[:,7]), np.mean(inner_results[:,8]), np.mean(inner_results[:,9]), np.mean(inner_results[:,10]), log
      
  
def safe_division(n, d):
    """
    Safely divides two numbers. Returns zero if d = 0.
    """
    return n / d if d else 0
    

def final_evaluation(model, thres, df_data, df_heldout, predictors, target, n_fold, log_path, c_ratio, imbalanced_positive, cost_function_param_name, resampling_method):
    """
    Performs both evaluation of validation or test sets.
    """
    data_x=df_data[predictors]
    data_y=df_data[target]
    
    if c_ratio != None:
        if imbalanced_positive:
            custom_class_ratio = {cost_function_param_name:{1:1,0:'%.2f' % (c_ratio)}}
        else:
            custom_class_ratio = {cost_function_param_name:{0:1,1:'%.2f' % (c_ratio)}}
        model.set_params(**custom_class_ratio)
    
    if resampling_method != None:
        data_x, data_y = resample_data(data_x=data_x, data_y=data_y, method=resampling_method)    
    
    if thres != None:
        acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(t=thres, alg=model, x_train=data_x, x_heldout=df_heldout[predictors], y_train=data_y, y_heldout=df_heldout[target])
    else:
        acc, tpr, tnr, ppv, npv, far, tss, f1, hss, auc, ap = fit_model(alg=model, x_train=data_x, x_heldout=df_heldout[predictors], y_train=data_y, y_heldout=df_heldout[target])
        
    log_entry = str('%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f' % (n_fold,acc,tpr,tnr,ppv,npv,far,tss,f1,hss,auc,ap))
    append_text_to_file(log_path, log_entry + '\n')    


def threshold_predict(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t). Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def univariate_feature_selection(k, data_x, data_y):
    """
    Returns k-best features based of F-score analysis.
    """
    df_kbest = SelectKBest(f_classif, k=k).fit(X=data_x, y=data_y)
    df_kbest_support = df_kbest.get_support()
    df_kbest_selected_features = []
    
    for bool, feature in zip(df_kbest_support, df_train.columns):
        if bool:
            df_kbest_selected_features.append(feature)
    
    return df_kbest_selected_features, pd.DataFrame(SelectKBest(f_classif, k=k).fit_transform(X=data_x, y=data_y), columns=df_kbest_selected_features)

       
help_process_param = str('OUT.KFD: Outer K-Fold; ' 
                          +'OUT.KFD.TS: Outer K-Fold (Time Separated Periods); '
                          +'MDL.SEL: Model Selection; '
                          +'FS.FLT: Univariate Filter Analysis; '
                          +'FS.WRP: Wrapper Analysis; '
                          +'HOP.GRD: Grid Analysis; '
                          +'HOP.VAL: Parameters Validation; '
                          +'DAT.RSP: Data Resampling; '
                          +'CST.FUN: Cost Function Analysis; '
                          +'CST.VAL: Cp/Cn Validation; '
                          +'CFP.ANL: Cut-Off Point Analysis; '
                          +'CFP.VAL: Cut-Off Point Validation; '
                          +'VAL.EVL: Validation Set Evaluation;'
                          +'TST.EVL: Test Set Evaluation.')

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--target", required=True, help="flare_t1do, flare_t2do, flare_t3do,...")
parser.add_argument("-p", "--process", required=True, help=help_process_param)
args = vars(parser.parse_args())
TARGET = args["target"].lower()
PROCESS = args["process"].lower()

if (TARGET == 'flare_t1do' 
    or TARGET == 'flare_t2do' 
    or TARGET == 'flare_t3do'): 
        pass
else: 
    print '[INFO] Unknown target!'
    exit()
    
if (PROCESS == 'out.kfd' 
    or PROCESS == 'out.kfd.ts' 
    or PROCESS == 'mdl.sel' 
    or PROCESS == 'fs.flt'
    or PROCESS == 'fs.wrp' 
    or PROCESS == 'hop.grd' 
    or PROCESS == 'hop.val' 
    or PROCESS == 'dat.rsp'
    or PROCESS == 'cst.fun'
    or PROCESS == 'cst.val'
    or PROCESS == 'cfp.anl'
    or PROCESS == 'cfp.val'
    or PROCESS == 'val.evl'
    or PROCESS == 'tst.evl'):
        pass
else: 
    print '[INFO] Unknown process!'
    exit()


PIPELINE = PROCESS
LOCK = multiprocessing.Lock()

###################################################
# Script Input Data ###############################

N_ITER = 20
MAG = 'm-x-flare' # c-m-x-flare
# DATA_FOLDER must adhere to the following format: /home/ubuntu/data/target.not-flare_m-x-flare/flare_t1do_m-x-flare/
DATA_FOLDER = '/home/ubuntu/data/target.not-flare_' + MAG + '/' + TARGET + '_' + MAG + '/'

PREDICTORS=[
    'radio_flux_10.7cm_t5', 'radio_flux_10.7cm_t4', 'radio_flux_10.7cm_t3',  'radio_flux_10.7cm_t2', 'radio_flux_10.7cm_t1',
    'sesc_sunspot_number_t5', 'sesc_sunspot_number_t4', 'sesc_sunspot_number_t3', 'sesc_sunspot_number_t2', 'sesc_sunspot_number_t1', 
    'sunspot_area_t5', 'sunspot_area_t4', 'sunspot_area_t3', 'sunspot_area_t2', 'sunspot_area_t1',
    'goes15_xray_bkgd_flux_t5', 'goes15_xray_bkgd_flux_t4', 'goes15_xray_bkgd_flux_t3', 'goes15_xray_bkgd_flux_t2', 'goes15_xray_bkgd_flux_t1',
    'z_component_wmfr_t5', 'z_component_wmfr_t4', 'z_component_wmfr_t3', 'z_component_wmfr_t2', 'z_component_wmfr_t1',
    'p_component_wmfr_t5', 'p_component_wmfr_t4', 'p_component_wmfr_t3', 'p_component_wmfr_t2', 'p_component_wmfr_t1',
    'c_component_wmfr_t5', 'c_component_wmfr_t4', 'c_component_wmfr_t3', 'c_component_wmfr_t2', 'c_component_wmfr_t1',
    'mag_type_wmfr_t5', 'mag_type_wmfr_t4', 'mag_type_wmfr_t3', 'mag_type_wmfr_t2', 'mag_type_wmfr_t1'
]

PREDICTORS_HEADER = str('radio_flux_10.7cm_t5;radio_flux_10.7cm_t4;radio_flux_10.7cm_t3;radio_flux_10.7cm_t2;radio_flux_10.7cm_t1;'
                        +'sesc_sunspot_number_t5;sesc_sunspot_number_t4;sesc_sunspot_number_t3;sesc_sunspot_number_t2;sesc_sunspot_number_t1;'
                        +'sunspot_area_t5;sunspot_area_t4;sunspot_area_t3;sunspot_area_t2;sunspot_area_t1;'
                        +'goes15_xray_bkgd_flux_t5;goes15_xray_bkgd_flux_t4;goes15_xray_bkgd_flux_t3;goes15_xray_bkgd_flux_t2;goes15_xray_bkgd_flux_t1;'
                        +'z_component_wmfr_t5;z_component_wmfr_t4;z_component_wmfr_t3;z_component_wmfr_t2;z_component_wmfr_t1;'
                        +'p_component_wmfr_t5;p_component_wmfr_t4;p_component_wmfr_t3;p_component_wmfr_t2;p_component_wmfr_t1;'
                        +'c_component_wmfr_t5;c_component_wmfr_t4;c_component_wmfr_t3;c_component_wmfr_t2;c_component_wmfr_t1;'
                        +'mag_type_wmfr_t5;mag_type_wmfr_t4;mag_type_wmfr_t3;mag_type_wmfr_t2;mag_type_wmfr_t1;'
                        +TARGET)

BASELINE_MODELS = ['baseline-forest', 'baseline-gbm', 'baseline-ada']

PARAM_GRID_GBM={
    'n_estimators':range(50, 501, 50),
    'max_depth':range(5,16,2),
    'min_samples_split':range(50,1001,50),
    'min_samples_leaf':range(5,101,5),
    'max_features':range(7,11,1),
    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
    'learning_rate':[0.1,0.01,0.05,0.5]
}
PARAM_GRID_ADA={
    'n_estimators':range(50, 501, 50),
    'base_estimator__max_depth':range(5,16,2),
    'base_estimator__min_samples_split':range(50,1001,50),
    'base_estimator__min_samples_leaf':range(5,101,5),
    'base_estimator__max_features':range(1,30,1),
    'learning_rate':[0.1,0.01,0.05,0.5]
}
PARAM_GRID_FOREST={
    'n_estimators':range(50, 501, 50),
    'max_depth':range(5,16,1),
    'min_samples_split':range(50,1001,50),
    'min_samples_leaf':range(5,101,5),
    'max_features':range(1,5,1),
    'min_impurity_split':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
}

# 'adasyn', 'random-over-sampler', 'smote', 'smote-tomek', 'edited-nn', 'repeated-edited-nn', 
# 'all-knn', 'one-sided-selection', 'cluster-centroids', 'random-under-sampler', 'neighbourhood-cleaning-rule',
# 'condensed-nearest-neighbour', 'near-miss', 'instance-hardness-threshold'
RESAMPLING_METHODS=['smote-enn', 'smote-tomek', 'smote']

# T1D Set-Up ######################################
BASELINE_MODEL_T1D = None
BASELINE_MODEL_PARAM_GRID_T1D = None
OPTIMUM_MODEL_T1D = None
PREDICTORS_SUBSET_T1D = [None]
MAX_FEATURES_PARAM_NAME_T1D = None
C_RATIO_T1D = None
COST_FUNCTION_PARAM_NAME_T1D = None
RESAMPLING_METHOD_T1D = None
PREDICTION_THRESHOLD_T1D = None
IMBALANCED_POSITIVE_T1D = None

# T2D Set-Up ######################################
BASELINE_MODEL_T2D = None
BASELINE_MODEL_PARAM_GRID_T2D = None
OPTIMUM_MODEL_T2D = None
PREDICTORS_SUBSET_T2D = [None]
MAX_FEATURES_PARAM_NAME_T2D = None
C_RATIO_T2D = None
COST_FUNCTION_PARAM_NAME_T2D = None
RESAMPLING_METHOD_T2D = None
PREDICTION_THRESHOLD_T2D = None
IMBALANCED_POSITIVE_T2D = None

# T3D Set-Up ######################################
BASELINE_MODEL_T3D = None
BASELINE_MODEL_PARAM_GRID_T3D = None
OPTIMUM_MODEL_T3D = None
PREDICTORS_SUBSET_T3D = [None]
MAX_FEATURES_PARAM_NAME_T3D = None
C_RATIO_T3D = None
COST_FUNCTION_PARAM_NAME_T3D = None
RESAMPLING_METHOD_T3D = None
PREDICTION_THRESHOLD_T3D = None
IMBALANCED_POSITIVE_T3D = None

# End of Input Data ###############################
###################################################

if TARGET == 'flare_t1do':
    BASELINE_MODEL = BASELINE_MODEL_T1D
    BASELINE_MODEL_PARAM_GRID = BASELINE_MODEL_PARAM_GRID_T1D
    PREDICTORS_SUBSET = PREDICTORS_SUBSET_T1D
    OPTIMUM_MODEL = OPTIMUM_MODEL_T1D
    PREDICTION_THRESHOLD = PREDICTION_THRESHOLD_T1D
    COST_FUNCTION_PARAM_NAME = COST_FUNCTION_PARAM_NAME_T1D
    if OPTIMUM_MODEL != None:
        MAX_FEATURES = OPTIMUM_MODEL.get_params()[MAX_FEATURES_PARAM_NAME_T1D]
    if BASELINE_MODEL != None:
        BASELINE_MAX_FEATURES = BASELINE_MODEL.get_params()[MAX_FEATURES_PARAM_NAME_T1D]
    C_RATIO = C_RATIO_T1D
    RESAMPLING_METHOD = RESAMPLING_METHOD_T1D
    IMBALANCED_POSITIVE = IMBALANCED_POSITIVE_T1D
elif TARGET == 'flare_t2do':
    BASELINE_MODEL = BASELINE_MODEL_T2D
    BASELINE_MODEL_PARAM_GRID = BASELINE_MODEL_PARAM_GRID_T2D
    PREDICTORS_SUBSET = PREDICTORS_SUBSET_T2D
    OPTIMUM_MODEL = OPTIMUM_MODEL_T2D
    PREDICTION_THRESHOLD = PREDICTION_THRESHOLD_T2D
    COST_FUNCTION_PARAM_NAME = COST_FUNCTION_PARAM_NAME_T2D
    if OPTIMUM_MODEL != None:
        MAX_FEATURES = OPTIMUM_MODEL.get_params()[MAX_FEATURES_PARAM_NAME_T2D]
    if BASELINE_MODEL != None:
        BASELINE_MAX_FEATURES = BASELINE_MODEL.get_params()[MAX_FEATURES_PARAM_NAME_T2D]
    C_RATIO = C_RATIO_T2D
    RESAMPLING_METHOD = RESAMPLING_METHOD_T2D
    IMBALANCED_POSITIVE = IMBALANCED_POSITIVE_T2D
elif TARGET == 'flare_t3do':
    BASELINE_MODEL = BASELINE_MODEL_T3D
    BASELINE_MODEL_PARAM_GRID = BASELINE_MODEL_PARAM_GRID_T3D
    PREDICTORS_SUBSET = PREDICTORS_SUBSET_T3D
    OPTIMUM_MODEL = OPTIMUM_MODEL_T3D
    PREDICTION_THRESHOLD = PREDICTION_THRESHOLD_T3D
    COST_FUNCTION_PARAM_NAME = COST_FUNCTION_PARAM_NAME_T3D
    if OPTIMUM_MODEL != None:
        MAX_FEATURES = OPTIMUM_MODEL.get_params()[MAX_FEATURES_PARAM_NAME_T3D]
    if BASELINE_MODEL != None:
        BASELINE_MAX_FEATURES = BASELINE_MODEL.get_params()[MAX_FEATURES_PARAM_NAME_T3D]
    C_RATIO = C_RATIO_T3D
    RESAMPLING_METHOD = RESAMPLING_METHOD_T3D
    IMBALANCED_POSITIVE = IMBALANCED_POSITIVE_T3D

#################################################
# Pipeline out.kfd: Outer K-Fold ################
if PIPELINE == 'out.kfd':
    df_data = pd.read_csv(DATA_FOLDER + 'dataset.ft.' + TARGET + '_' + MAG + '.csv', sep=';', decimal=',')
    header = PREDICTORS_HEADER
    outer_data_split(header=header, 
                     save_path=DATA_FOLDER, 
                     data_x=df_data[PREDICTORS], 
                     data_y=df_data[TARGET], 
                     kf_splits=5, 
                     test_size=0.05, 
                     predictors=PREDICTORS, 
                     target=TARGET)
    
########################################################
# Pipeline out.kfd.ts: Outer K-Fold (Time Separated) ###
if PIPELINE == 'out.kfd.ts':
    df_data_train = pd.read_csv(DATA_FOLDER + 'dataset.ft.' + TARGET + '_' + MAG + '.train.csv', sep=';', decimal=',')
    df_data_test = pd.read_csv(DATA_FOLDER + 'dataset.ft.' + TARGET + '_' + MAG + '.test.csv', sep=';', decimal=',')
    header = PREDICTORS_HEADER
    outer_data_split_time_separated(header=header, 
                                    save_path=DATA_FOLDER, 
                                    df_data_train=df_data_train, 
                                    df_data_test=df_data_test, 
                                    kf_splits=5,
                                    predictors=PREDICTORS, 
                                    test_size=0.2,
                                    target=TARGET)
    
###################################################
# Pipeline mdl.sel: Model Selection ###############
elif PIPELINE == 'mdl.sel':
    log_path = DATA_FOLDER + "out/mdl.sel." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='MODEL;N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    procs = [] 
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_spot_check_models(models=BASELINE_MODELS, 
                                      df_train=df_train, 
                                      predictors=PREDICTORS,
                                      target=TARGET, 
                                      n_splits=5, 
                                      n_iterations=N_ITER, 
                                      n_fold=n_fold, 
                                      log_path=log_path,
                                      lock=LOCK)
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##################################################
# Pipeline fs.flt: Filter Analysis ###############
elif PIPELINE == 'fs.flt':
    log_path = DATA_FOLDER +"out/fs.flt." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    procs = []    
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_univariate_feature_analysis(model=BASELINE_MODEL, 
                                                data_x=df_train[PREDICTORS], 
                                                data_y=df_train[TARGET],
                                                n_splits=5, 
                                                n_iterations=N_ITER, 
                                                n_fold=n_fold, 
                                                log_path=log_path,
                                                max_features=BASELINE_MAX_FEATURES, 
                                                lock=LOCK)
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##################################################
# Pipeline fs.wrp: Wrapper Analysis ##############
elif PIPELINE == 'fs.wrp':
    log_path = DATA_FOLDER + "out/fs.wrp." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    procs = []  
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_wrapper_analysis(model=BASELINE_MODEL, 
                                     df_train=df_train, 
                                     n_splits=5, 
                                     n_iterations=N_ITER, 
                                     n_fold=n_fold,     
                                     log_path=log_path, 
                                     predictors_subset=PREDICTORS_SUBSET,
                                     target=TARGET, 
                                     lock=LOCK)
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##################################################
# Pipeline hop.grd: Grid Analysis ################
elif PIPELINE == 'hop.grd':
    log_path = DATA_FOLDER + "out/hop.grd." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    for n_fold in range(1,6):
        print("\nworking on n_fold=" + str(n_fold) + "\n")
        log_entry='n_fold: ' + str(n_fold)
        append_text_to_file(log_path, log_entry + '\n')
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        randomized_search_cv(data_x=df_train[PREDICTORS_SUBSET], 
                             data_y=df_train[TARGET], 
                             param_grid=BASELINE_MODEL_PARAM_GRID, 
                             model=BASELINE_MODEL,                 
                             n_scores=1, 
                             n_folds=5, 
                             n_iter=N_ITER,
                             scoring='roc_auc',
                             verbose=True,
                             log_path=log_path)

#################################################
# Pipeline hop.val: Parameters Validation #######
elif PIPELINE == 'hop.val':
    log_path = DATA_FOLDER + "out/hop.val." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    procs = []
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_parameters_validation(model=OPTIMUM_MODEL, 
                                          df_train=df_train, 
                                          n_splits=5, 
                                          n_iterations=N_ITER, 
                                          n_fold=n_fold, 
                                          predictors_subset=PREDICTORS_SUBSET,
                                          target=TARGET,
                                          log_path=log_path,
                                          lock=LOCK)
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##########################################
# Pipeline dat.rsp: Data Resampling ######
elif PIPELINE == 'dat.rsp':
    log_path = DATA_FOLDER + "out/dat.rsp." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='METHOD;N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    procs = []  
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_data_resampling(model=OPTIMUM_MODEL, 
                                    methods=RESAMPLING_METHODS,
                                    df_train=df_train, 
                                    predictors=PREDICTORS_SUBSET, 
                                    target=TARGET,
                                    n_splits=5, 
                                    n_iterations=N_ITER, 
                                    n_fold=n_fold,
                                    log_path=log_path,
                                    lock=LOCK) 
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##################################################
# Pipeline cst.fun: Cp/Cn Analysis ###############
elif PIPELINE == 'cst.fun':
    log_path = DATA_FOLDER + "out/cst.fun." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')       
    procs = []  
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_cost_function_analysis(model=OPTIMUM_MODEL, 
                                           predictors=PREDICTORS_SUBSET, 
                                           target=TARGET, 
                                           df_train=df_train,
                                           n_splits=5, 
                                           n_iterations=N_ITER,
                                           r_start=0.05,
                                           r_end=1.0,
                                           r_step=0.025,
                                           n_fold=n_fold,     
                                           log_path=log_path,
                                           imbalanced_positive=IMBALANCED_POSITIVE,
                                           cost_function_param_name=COST_FUNCTION_PARAM_NAME,
                                           lock=LOCK)
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
#################################################
# Pipeline cst.val: Cost Function Validation ####
elif PIPELINE == 'cst.val':
    log_path = DATA_FOLDER + "out/cst.val." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    procs = []  
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_cost_function_validation(model=OPTIMUM_MODEL,  
                                             df_train=df_train,
                                             n_splits=5,   
                                             n_iterations=N_ITER, 
                                             n_fold=n_fold,     
                                             log_path=log_path,
                                             predictors=PREDICTORS_SUBSET,
                                             c_ratio=C_RATIO,
                                             target=TARGET,
                                             cost_function_param_name=COST_FUNCTION_PARAM_NAME,
                                             imbalanced_positive=IMBALANCED_POSITIVE,
                                             lock=LOCK)  
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##################################################
# Pipeline cfp.anl: t Analysis ###################        
elif PIPELINE == 'cfp.anl':
    log_path = DATA_FOLDER + "out/cfp.anl." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    procs = []  
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_cut_off_point_analysis(model=OPTIMUM_MODEL, 
                                           df_train=df_train, 
                                           target=TARGET, 
                                           predictors=PREDICTORS_SUBSET,
                                           n_splits=5, 
                                           n_iterations=N_ITER,
                                           t_start=0.025,
                                           t_end=1.0,
                                           t_step=0.025,
                                           n_fold=n_fold,     
                                           log_path=log_path,
                                           cost_function_param_name=COST_FUNCTION_PARAM_NAME,
                                           c_ratio=C_RATIO,
                                           resampling_method=RESAMPLING_METHOD,
                                           imbalanced_positive=IMBALANCED_POSITIVE,
                                           lock=LOCK)    
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
##################################################
# Pipeline cfp.val: cut off point validation #####
elif PIPELINE == 'cfp.val':
    log_path = DATA_FOLDER + "out/cfp.val." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    procs = [] 
    for n_fold in range(1,6):
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        p = process_cut_off_point_validation(model=OPTIMUM_MODEL, 
                                             predictors=PREDICTORS_SUBSET, 
                                             target=TARGET, 
                                             df_train=df_train,
                                             t=PREDICTION_THRESHOLD,
                                             n_iterations=N_ITER,
                                             n_splits=5, 
                                             n_fold=n_fold,     
                                             log_path=log_path,
                                             cost_function_param_name=COST_FUNCTION_PARAM_NAME,
                                             c_ratio=C_RATIO,
                                             resampling_method=RESAMPLING_METHOD,
                                             imbalanced_positive=IMBALANCED_POSITIVE,
                                             lock=LOCK)
        procs.append(p)
    for i in range(5):
        procs[i].start()
    for i in range(5):
        procs[i].join()
        
################################################
# Pipeline val.evl: Validation Set Evaluation ##
elif PIPELINE == 'val.evl':
    log_path = DATA_FOLDER + "out/val.evl.raw." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    for n_fold in range(1,6):
        print("\nworking on n_fold=" + str(n_fold) + "\n")
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        df_val = pd.read_csv(DATA_FOLDER + 'data_val.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')        
        final_evaluation(model=BASELINE_MODEL, 
                         thres=None, 
                         df_data=df_train, 
                         df_heldout=df_val, 
                         predictors=PREDICTORS, 
                         target=TARGET, 
                         n_fold=n_fold, 
                         log_path=log_path,
                         cost_function_param_name=None,
                         c_ratio=None,
                         resampling_method=None,
                         imbalanced_positive=None)
    log_path = DATA_FOLDER + "out/val.evl.framework." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS'
    append_text_to_file(log_path, log_entry + '\n')
    for n_fold in range(1,6):
        print("\nworking on n_fold=" + str(n_fold) + "\n")
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        df_val = pd.read_csv(DATA_FOLDER + 'data_val.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')        
        final_evaluation(model=OPTIMUM_MODEL, 
                         thres=PREDICTION_THRESHOLD, 
                         df_data=df_train, 
                         df_heldout=df_val, 
                         predictors=PREDICTORS_SUBSET, 
                         target=TARGET, 
                         n_fold=n_fold, 
                         log_path=log_path,
                         cost_function_param_name=COST_FUNCTION_PARAM_NAME,
                         c_ratio=C_RATIO,
                         resampling_method=RESAMPLING_METHOD,
                         imbalanced_positive=IMBALANCED_POSITIVE)
        
##########################################
# Pipeline tst.evl: Test Set Evaluation ##
elif PIPELINE == 'tst.evl':
    log_path = DATA_FOLDER+"out/tst.evl.raw." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS;AUC;AP'
    append_text_to_file(log_path, log_entry + '\n')
    for n_fold in range(1,6):
        print("\nworking on n_fold=" + str(n_fold) + "\n")
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        df_test = pd.read_csv(DATA_FOLDER + 'data_test.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')        
        final_evaluation(model=BASELINE_MODEL, 
                        thres=None, 
                        df_data=df_train, 
                        df_heldout=df_test, 
                        predictors=PREDICTORS, 
                        target=TARGET, 
                        n_fold=n_fold, 
                        log_path=log_path,
                        cost_function_param_name=None,
                        c_ratio=None,
                        resampling_method=None,
                        imbalanced_positive=None)
    log_path = DATA_FOLDER + "out/tst.evl.framework." + TARGET
    if os.path.exists(log_path): 
        key = int(input('[INPUT] ' + log_path + ' already exists... overwrite? [1] - yes; [2] - no: '))
        if (key == 1):
           os.remove(log_path)
           print('[INFO] ' + log_path + ' deleted... a new one will be created.') 
        elif (key == 2): print('[INFO] ' + log_path + ' will not be deleted and thus was opened in append mode.')
        else: print ('[INFO] invalid option... ' + log_path + ' will be opened in append mode.')
    else: print('[INFO] ' + log_path + ' does not exist, so it will be created.')
    log_entry='N_FOLD;ACC;TPR;TNR;PPV;NPV;FAR;TSS;F1;HSS'
    append_text_to_file(log_path, log_entry + '\n')
    for n_fold in range(1,6):
        print("\nworking on n_fold=" + str(n_fold) + "\n")
        df_train = pd.read_csv(DATA_FOLDER + 'data_train.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')
        df_test = pd.read_csv(DATA_FOLDER + 'data_test.fold.' + str(n_fold) + '.csv', sep=';', decimal='.')        
        final_evaluation(model=OPTIMUM_MODEL, 
                        thres=PREDICTION_THRESHOLD, 
                        df_data=df_train, 
                        df_heldout=df_test, 
                        predictors=PREDICTORS_SUBSET, 
                        target=TARGET, 
                        n_fold=n_fold, 
                        log_path=log_path,
                        cost_function_param_name=COST_FUNCTION_PARAM_NAME,
                        c_ratio=C_RATIO,
                        resampling_method=RESAMPLING_METHOD,
                        imbalanced_positive=IMBALANCED_POSITIVE)
