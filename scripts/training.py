#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:51:24 2017

@author: faezeh.salehi
"""

# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function
import optparse
import csv
import json
import os
import time
import sys
import pickle
import base64
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold


def one_hot_dataframe_train(data, cols, replace=True):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns the data and the fitted vectorizor.
    """
    vec = DictVectorizer(dtype=np.int8)
    vecData = vec.fit_transform(data[cols].to_dict(orient='records')).toarray()
    feature_names = vec.get_feature_names()

    return vecData, vec, np.array(feature_names)

    
    
def preprocess(dataframe):

    #start = time.time()
    #filling columns with all missing values with 0
    for col in dataframe.columns[pd.isnull(dataframe).all()]:
        dataframe[col] = dataframe[col].astype(object).fillna(0)
    
    dataframe.pop('acquisition_cost')
    dataframe.pop('cb_campaign')
    dataframe.pop('advertiser_app')
    dataframe.pop('device_id')
    dataframe.pop('max_install_ts')
    dataframe.pop('install_dt')
    dataframe.pop('pia_install')
    dataframe.pop('first_purchase_dt')
    dataframe.pop('last_seen')
    
    
    #filling missing numeric values with mean and non-numeric with "unknown
    dataframe[["d7_iap_count", "d7_iap_revenue","l7_retention", "installs","d7_ad_revenue", "bootups",
               "iap_count","iap_spend","cat_action","cat_adventure","cat_card","cat_casino",
               "cat_educational","cat_family","cat_music","cat_non_game","cat_puzzle","cat_racing",
               "cat_role_playing","cat_simulation","cat_sports","cat_strategy","cat_trivia","cat_other",
               "affinity"]] = dataframe[["d7_iap_count", "d7_iap_revenue","l7_retention", "installs","d7_ad_revenue", "bootups",
               "iap_count","iap_spend","cat_action","cat_adventure","cat_card","cat_casino",
               "cat_educational","cat_family","cat_music","cat_non_game","cat_puzzle","cat_racing",
               "cat_role_playing","cat_simulation","cat_sports","cat_strategy","cat_trivia","cat_other",
               "affinity"]].fillna(0)
    
    dataframe[["country", "platform", "model", "campaign_type", "publisher_app",   
               "os", "model_id", "language", "region", "city", "dma"]] = dataframe[["country", "platform", "model", "campaign_type", "publisher_app",  
               "os", "model_id", "language", "region", "city", "dma"]].fillna("unknown")
    
    #copy dataframe
    xtrain = dataframe.copy()

    #d7_iap_count
    d7_iap_count = xtrain.pop('d7_iap_count')
    
    #d7_iap_revenue
    d7_iap_revenue = xtrain.pop('d7_iap_revenue')
    
    #d7_ad_revenue
    d7_ad_revenue = xtrain.pop('d7_ad_revenue')
    
    #target value
    ytrain = pd.to_numeric(xtrain.pop('l7_retention').values)

    #multiple options for target value
    ytrain1=ytrain.copy()
    ytrain2=ytrain.copy()
    ytrain3=ytrain.copy()
    ytrain4=ytrain.copy()
    
    ytrain1[ytrain1>0]=1
    ytrain2[ytrain2<=1]=0
    ytrain2[ytrain2>1]=1
    ytrain3[ytrain3<=2]=0
    ytrain3[ytrain3>2]=1
    ytrain4[ytrain4<=3]=0
    ytrain4[ytrain4>3]=1
    
    
    
    #add sample weights to balance data
    #sample_weights = np.array([float((len(ytrain[ytrain==1])+1))/(len(ytrain[ytrain==0]+1)) if i == 0 else 1 for i in ytrain])
   
    #convert to lower case
    xtrain = xtrain.apply(lambda x: x.astype(str).str.lower())
    
    #convert to numerical
    xtrain[["installs", "bootups","iap_count","iap_spend","cat_action","cat_adventure","cat_card","cat_casino",
               "cat_educational","cat_family","cat_music","cat_non_game","cat_puzzle","cat_racing",
               "cat_role_playing","cat_simulation","cat_sports","cat_strategy","cat_trivia","cat_other",
               "affinity"]] = xtrain[["installs","bootups", "iap_count","iap_spend","cat_action","cat_adventure","cat_card","cat_casino",
               "cat_educational","cat_family","cat_music","cat_non_game","cat_puzzle","cat_racing",
               "cat_role_playing","cat_simulation","cat_sports","cat_strategy","cat_trivia","cat_other",
               "affinity"]].apply(pd.to_numeric)

    #convert categorical variables to binary variables
    xtrain_Arr, vec, feature_names = one_hot_dataframe_train(xtrain, ["country", "platform", "model", "campaign_type", "publisher_app",   
               "os", "model_id", "language", "region", "city", "dma"])    
    
    #print "finished one hot"

    #xtrain_new = np.empty((len(xtrain_Arr),0))
    #selected_features = []

    #for i in range(0, len(feature_names), 1000):
    #    try:
    #        sel = VarianceThreshold(threshold=(thresh))
    #        xtrain_batch = sel.fit_transform(xtrain_Arr[:,i:i+1000])
    #        features_batch = feature_names[i:i+1000][sel.get_support()]
    #        xtrain_new = np.append(xtrain_new, xtrain_batch, axis = 1)
    #        selected_features = np.append(selected_features, features_batch)
    #    except:
    #        continue

    xtrain_Arr = np.append(xtrain_Arr, np.array(xtrain[["installs", "bootups","iap_count","iap_spend","cat_action","cat_adventure","cat_card","cat_casino",
               "cat_educational","cat_family","cat_music","cat_non_game","cat_puzzle","cat_racing",
               "cat_role_playing","cat_simulation","cat_sports","cat_strategy","cat_trivia","cat_other",
               "affinity"]],dtype='float64'),axis=1)
        
    #selected_features = np.append(selected_features,['device_advertising_app_impressions','bid_value'])

    #print "done sel_names : %s" %len(sel_names)
    #sel_dict = {}
    
    #for s in selected_features:
    #    sel_dict[s] = len(sel_dict)        

    #min max scaling
    #min_max_scaler = MinMaxScaler()
    #xtrain_minmax = min_max_scaler.fit_transform(xtrain_new)
    #scale = min_max_scaler.scale_
    #minim = min_max_scaler.min_
    
    return xtrain_Arr, ytrain, ytrain1, ytrain2, ytrain3, ytrain4

    
#grid search with cross-validation for param tuning
def GridSearchModel(Xtrain, ytrain, Xtest, ytest, model_name, input_dir):
    
    # Split the dataset train, test
    X_train, X_test, y_train, y_test = train_test_split(
        Xtrain, ytrain, test_size=0.3, random_state=0)
    
    scores = ['roc_auc', 'precision', 'recall']
    
    if model_name == "RF":
        param_grid = {'n_estimators' : [500],
                      'max_depth': [10, 5, 3],
                      'max_features': ['sqrt'],
                      'min_samples_split': [10, 5]
                     }
        clfs = []                    
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5,
                               scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()    
            clfs.append(clf)
            rf_gsm_pred = clf.predict(Xtest)
            rf_gsm_report = classification_report(ytest, rf_gsm_pred)
            
            outfile = open('%s_gs_%s_%s_model_report.txt' %(input_dir, model_name, score), 'wb')
            outfile.write("best parameters found: \n")
            outfile.write("%s \n" %clf.best_params_)
            outfile.write("%s \n" %rf_gsm_report)
            outfile.close()

        
    elif model_name == "GBC":
        param_grid = {'n_estimators' : [50, 200],
                      'max_depth': [10, 5, 3],
                      'min_samples_split': [10, 5]
                     }
        
        clfs = []                  
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5,
                               scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print() 
            clfs.append(clf)
            gb_gsm_pred = clf.predict(Xtest)
            gb_gsm_report = classification_report(ytest, gb_gsm_pred)
            
            outfile = open('%s_gs_%s_%s_model_report.txt' %(input_dir, model_name, score), 'wb')
            outfile.write("best parameters found: \n")
            outfile.write("%s \n" %clf.best_params_)
            outfile.write("%s \n" %gb_gsm_report)
            outfile.close()
            
            

        
    elif model_name == "SGC":
        print("place holder")
        
    else:
        sys.stderr.write("unexpected model name!")
        exit(1)   
    
    return clfs



#gradient boosting classifier training
def gb_classify(X, Y):
    gbc = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3)
    #gbc.fit(X, Y, sample_weight=sample_weights)
    gbc.fit(X, Y)
    return gbc

#gradient boosting regressor
def gb_regressor(X, Y):
    gbr = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=100,max_depth=3)
    #gbr.fit(X, Y, sample_weight=sample_weights)
    gbr.fit(X, Y)
    return gbr


#random forest classifier
def rf_classify(X, Y):
    rf = ensemble.RandomForestClassifier(n_estimators=10)
    rf.fit(X, Y)
    return rf


#stochastic gradient descent classifier
def SG_classify(X, Y, sgc=None):
    if sgc:
        if np.bincount(Y)[0]>0 and len(np.bincount(Y))>1:
            sgc.partial_fit(X, Y)
    else:
        #param_SGC = {'loss': 'hinge', 'penalty': 'elasticnet', 'n_iter': 1, 'shuffle': True, 'class_weight': {0: class_0_weight, 1:class_1_weight}, 'warm_start':True, 'alpha':0.001}

        param_SGC = {'loss': 'hinge', 'penalty': 'elasticnet', 'n_iter': 1, 'shuffle': True, 'warm_start':True, 'alpha':0.001}

        sgc = SGDClassifier(**param_SGC)
        if np.bincount(Y)[0]>0 and len(np.bincount(Y))>1:
            sgc.partial_fit(X, Y, np.unique(Y))
            coef = sgc.coef_
            intercept = sgc.intercept_
        else:
            sgc=None
            coef = None
            intercept = None 
        
    return sgc, coef, intercept

    
    
def main():

    #variance_threshold = 0.01
    
    parser = optparse.OptionParser()
    parser.add_option("-i", "--input_file", type="string", dest="input_file",
                      help="path to the training input file, eg, data.csv",
                      default="../data/5374e872c26ee411068adbee")

    
    (opts, args) = parser.parse_args()

    input_dir = opts.input_file.split(".csv")[0]

    columns = ["advertiser_app", "device_id", "country", "platform", "model", "campaign_type", 
               "cb_campaign","acquisition_cost","publisher_app","pia_install", "d7_iap_revenue",
               "d7_iap_count", "first_purchase_dt", "d7_ad_revenue", "l7_retention", "os",
               "model_id", "language", "region", "city", "dma", "last_seen", "installs", "max_install_ts",
               "bootups","iap_count","iap_spend","cat_action","cat_adventure","cat_card","cat_casino",
               "cat_educational","cat_family","cat_music","cat_non_game","cat_puzzle","cat_racing",
               "cat_role_playing","cat_simulation","cat_sports","cat_strategy","cat_trivia","cat_other","affinity",
               "install_dt"]



    #load data
    training_all=pd.read_csv("%s" %opts.input_file, header=0, sep='\t', dtype={"advertiser_app":np.str, "device_id":np.str, "country":np.str, "platform":np.str, "model":np.str, 
               "campaign_type":np.str, "cb_campaign":np.bool,"acquisition_cost":np.float64,"publisher_app":np.str,"pia_install":np.float64, "d7_iap_revenue":np.float64,
               "d7_iap_count":np.int64, "first_purchase_dt":np.str, "d7_ad_revenue":np.float64, "l7_retention":np.float64, "os":np.str,
               "model_id":np.str, "language":np.str, "region":np.str, "city":np.str, "dma":np.str, "last_seen":np.float64, "installs":np.float64, "max_install_ts":np.float64,
               "bootups":np.float64,"iap_count":np.float64,"iap_spend":np.float64,"cat_action":np.float64,"cat_adventure":np.float64,"cat_card":np.float64,"cat_casino":np.float64,
               "cat_educational":np.float64,"cat_family":np.float64,"cat_music":np.float64,"cat_non_game":np.float64,"cat_puzzle":np.float64,"cat_racing":np.float64,
               "cat_role_playing":np.float64,"cat_simulation":np.float64,"cat_sports":np.float64,"cat_strategy":np.float64,"cat_trivia":np.float64,"cat_other":np.float64,"affinity":np.float64,
               "install_dt":np.str}, names=columns, na_values="\\N")


    
    #preprocess the data and save the selected features in a pickle file
    X, y, y1, y2, y3, y4 = preprocess(training_all)
    
    #MyFeatures = open('%s_selected_features.pickle' %input_dir, 'wb') 
    #pickle.dump(sel_feature_dict, MyFeatures)
    #MyFeatures.close()
            
    indices = range(len(y1))
    
    #split data into training and testing
    Xtrain, Xtest, ytrain, ytest, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.3, random_state=0)

    #multiple target value splits
    ytrain1 = y1[indices_train]
    ytest1 = y1[indices_test]

    ytrain2 = y2[indices_train]
    ytest2 = y2[indices_test]
    
    ytrain3 = y3[indices_train]
    ytest3 = y3[indices_test]

    ytrain4 = y4[indices_train]
    ytest4 = y4[indices_test]

    #class_0_weight = len(ytrain)/(2.*np.bincount(ytrain)[0]+1)
    #class_1_weight = 1.3 * len(ytrain)/(2.*np.bincount(ytrain)[1]+1)
    
    #random forest classifier from grid search cv
    rf2_gsm = GridSearchModel(Xtrain, ytrain2, Xtest, ytest2, "RF", input_dir)

    #ranfom forest classifier                       
    rf2 = rf_classify(Xtrain, ytrain2)
    rf2_pred = rf2.predict(Xtest) 
    rf2_report = classification_report(ytest2, rf2_pred)
    outfile = open('%s_rf_model_report.txt' %(input_dir), 'wb')
    outfile.write("%s \n" %rf2_report)
    outfile.close()
    
    
    #gradient boosting classifier for different target variables
    gbc1 = gb_classify(Xtrain, ytrain1)
    gbc1_pred = gbc1.predict(Xtest) 
    gbc1_report = classification_report(ytest1, gbc1_pred)
    outfile = open('%s_gbc1_model_report.txt' %(input_dir), 'wb')
    outfile.write("%s \n" %gbc1_report)
    outfile.close()
    
    gbc2 = gb_classify(Xtrain, ytrain2)
    gbc2_pred = gbc2.predict(Xtest) 
    gbc2_report = classification_report(ytest2, gbc2_pred)
    outfile = open('%s_gbc2_model_report.txt' %(input_dir), 'wb')
    outfile.write("%s \n" %gbc2_report)
    outfile.close()
    
    gbc3 = gb_classify(Xtrain, ytrain3)
    gbc3_pred = gbc3.predict(Xtest) 
    gbc3_report = classification_report(ytest3, gbc3_pred)
    outfile = open('%s_gbc3_model_report.txt' %(input_dir), 'wb')
    outfile.write("%s \n" %gbc3_report)
    outfile.close()
    
    gbc4 = gb_classify(Xtrain, ytrain4)
    gbc4_pred = gbc4.predict(Xtest) 
    gbc4_report = classification_report(ytest4, gbc4_pred)
    outfile = open('%s_gbc4_model_report.txt' %(input_dir), 'wb')
    outfile.write("%s \n" %gbc4_report)
    outfile.close()
    
    #gradient boosting regression 
    gbr = gb_regressor(Xtrain, ytrain)
    gbr_pred = gbr.predict(Xtest) 
    gbr_report = metrics.mean_squared_error(ytest, gbr_pred)
    outfile = open('%s_gbr_model_report.txt' %(input_dir), 'wb')
    outfile.write("%s \n" %gbr_report)
    outfile.close()
    
 
    #pickle the trained classifier
    #MyModel = open('%s_model.pickle' %input_dir, 'wb') 
    #pickle.dump(XXX, MyModel)
    #MyModel.close() 
    
    #just do the ranking of the test set
    #ypred_prob = rf.predict_proba(Xtest)


    return


if __name__=="__main__":
    main()

    
    
    

        
        