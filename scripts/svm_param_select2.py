#!/usr/bin/env python
import os
import h5py
import json
import click
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

## needed with slurm to see local python library under working dir
#import sys
#sys.path.append(os.path.join(os.getcwd(), 'code'))
#
#import models
#from models import Base, User, Collection, Sample, Micrograph, dbpath
#from sqlalchemy import create_engine
#from sqlalchemy.orm import sessionmaker
#
#engine = create_engine('sqlite:///data/microstructures.sqlite')
#Base.metadata.bind = engine
#DBSession = sessionmaker(bind=engine)
#db = DBSession()

def load_representations(datafile):
    # grab image representations from hdf5 file
    keys, features = [], []

    with h5py.File(datafile, 'r') as f:
        for key in f:
            keys.append(key)
            features.append(f[key][...])

    return np.array(keys), np.array(features)

def select_balanced_dataset(labels, X, n_per_class=50, seed=0):
    """ select a balanced dataset for cross-validation """
    np.random.seed(seed) # set seed to enable deterministic training set
    selection = []
    tlabel = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        sel = np.random.choice(idx, n_per_class, replace=False)
        selection.append(sel)
        tlabel += [label]*n_per_class

    selection = np.concatenate(selection)
    l = np.array(tlabel)
    X = X[selection]
    np.random.seed() # reset seed to randomize cv folds across runs
    return l, X, selection

def cv_loop_rf(labels, X, cv, C=10, n_repeats=1, reduce_dim=None):
    
    # For random forest, C = number of trees
    
    tscore, vscore = [], []
    clf = RandomForestClassifier(n_estimators=C,
                                 class_weight='balanced')
    
    for repeat in range(n_repeats):
        for train, test in cv.split(X, labels):
            if reduce_dim:
                pca = PCA(n_components=reduce_dim).fit(X[train])
                Xtrain = pca.transform(X[train])
                Xtest = pca.transform(X[test])
            else:
                Xtrain = X[train]
                Xtest = X[test]
            
            print('train shape')
            print(Xtrain.shape)
            print('test shape')
            print(Xtest.shape)
            
            # L2-normalize instances for linear SVM following Vedaldi and Zisserman
            # Efficient Additive Kernels via Explicit Feature Maps
            # Also: Perronnin, Sanchez, and Mensink
            # Improving the Fisher kernel for large-scale image classification

            scaling = MinMaxScaler(feature_range=(-1,1)).fit(Xtrain)
            Xtrain = scaling.transform(Xtrain)
            Xtest = scaling.transform(Xtest)
            #            Xtrain = Xtrain / np.linalg.norm(Xtrain, axis=1)[:,np.newaxis]
            #            Xtest = Xtest / np.linalg.norm(Xtest, axis=1)[:,np.newaxis]

            clf.fit(Xtrain, labels[train])
            tscore.append(clf.score(Xtrain, labels[train]))
            print('train score')
            print(tscore)
            vscore.append(clf.score(Xtest, labels[test]))
            print('validate score')
            print(vscore)

            # Confusion Matrix
            predicted=clf.predict(Xtest)
            confusion = confusion_matrix(labels[test],predicted)
            print('confusion!')
            print(confusion)

    print('{} +/- {}'.format(np.mean(vscore), np.std(vscore, ddof=1)))
    return np.mean(vscore), np.std(vscore, ddof=1), np.mean(tscore), np.std(tscore, ddof=1)


def cv_loop_chi2(labels, X, cv, C=1, n_repeats=1):
    tscore, vscore = [], []
    for repeat in range(n_repeats):
        for train, test in cv.split(X, labels):
            # follow Zhang et al (2007) in setting gamma
            gamma = -1 / np.mean(additive_chi2_kernel(X[train]))
            clf = SVC(kernel=chi2_kernel, gamma=gamma, C=C,
                      class_weight='balanced', decision_function_shape='ovr', cache_size=2048)
    
            clf.fit(X[train], labels[train])
            tscore.append(clf.score(X[train], labels[train]))
            vscore.append(clf.score(X[test], labels[test]))

    print('{} +/- {}'.format(np.mean(vscore), np.std(vscore, ddof=1)))
    return np.mean(vscore), np.std(vscore, ddof=1), np.mean(tscore), np.std(tscore, ddof=1)

def cv_loop_linear(labels, X, cv, C=1, n_repeats=1, reduce_dim=None):
    tscore, vscore = [], []
    clf = SVC(kernel='linear', C=C,
              class_weight='balanced', decision_function_shape='ovr', cache_size=7000)
    for repeat in range(n_repeats):
        for train, test in cv.split(X, labels):

            if reduce_dim:
                pca = PCA(n_components=reduce_dim).fit(X[train])
                Xtrain = pca.transform(X[train])
                Xtest = pca.transform(X[test])
            else:
                Xtrain = X[train]
                Xtest = X[test]
            print('train shape')
            print(Xtrain.shape)
            print('test shape')
            print(Xtest.shape)
            # L2-normalize instances for linear SVM following Vedaldi and Zisserman
            # Efficient Additive Kernels via Explicit Feature Maps
            # Also: Perronnin, Sanchez, and Mensink
            # Improving the Fisher kernel for large-scale image classification

            scaling = MinMaxScaler(feature_range=(-1,1)).fit(Xtrain)
            Xtrain = scaling.transform(Xtrain)
            Xtest = scaling.transform(Xtest)
#            Xtrain = Xtrain / np.linalg.norm(Xtrain, axis=1)[:,np.newaxis]
#            Xtest = Xtest / np.linalg.norm(Xtest, axis=1)[:,np.newaxis]

            clf.fit(Xtrain, labels[train])
            tscore.append(clf.score(Xtrain, labels[train]))
            print('train score')
            print(tscore)
            vscore.append(clf.score(Xtest, labels[test]))
            print('validate score')
            print(vscore)

    print('{} +/- {}'.format(np.mean(vscore), np.std(vscore, ddof=1)))
    return np.mean(vscore), np.std(vscore, ddof=1), np.mean(tscore), np.std(tscore, ddof=1)

@click.command()
@click.argument('datafile', type=click.Path())
@click.argument('dbcsv', type=click.Path())
@click.argument('task', type=click.Path())
@click.option('--kernel', '-k', type=click.Choice(['linear', 'chi2', 'rbf', 'rf']), default='linear')
@click.option('--margin-param', '-C', type=float, default=None)
@click.option('--n-per-class', '-n', type=int, default=None)
@click.option('--n-repeats', '-r', type=int, default=1)
@click.option('--reduce-dim', '-d', type=int, default=None)
@click.option('--seed', '-s', type=int, default=None)
def svm_param_select(datafile, dbcsv, task, kernel, margin_param, n_per_class, n_repeats, reduce_dim, seed):
    # datafile = './data/full/features/vgg16_block5_conv3-vlad-32.h5'
    
    # Load keys and image features
    keys, features = load_representations(datafile)

    # Generate "labels" from df_mg pandas dataframe. Task must be a valid column in the csv database
    df_mg = pd.read_csv(dbcsv)
    df_mg.set_index('id',inplace=True)
    labels=np.array(df_mg[task].loc[[int(s) for s in keys]])

    # Get a balanced dataset
    if n_per_class is not None:
        l, X, sel = select_balanced_dataset(labels, features, n_per_class=n_per_class, seed=seed)
    else:
        l = labels
        X = features
        sel = [i for i in range(len(keys))]

    # split datasets for K-fold cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    # cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1)

    # Save the results
    print(datafile)
    
    if margin_param is not None:
        resultsdir = 'svmresults'
    else:
        resultsdir = 'svm'

    results = {
        'training_set': list(keys[sel]),
        'kernel': kernel,
            'n_per_class': n_per_class,
            'seed': seed,
            'n_repeats': n_repeats,
            'cv_C': {}
        }

    resultsfile = datafile.replace('features', resultsdir).replace('.h5', '-{kernel}-{n_per_class}.json'.format(**results))
    
    try:
        os.makedirs(os.path.dirname(resultsfile))
    except FileExistsError:
        pass

    if margin_param is not None:
        C_range = [margin_param]
    else:
        C_range = np.logspace(-12, 2, 15, base=2)
        
    for C in C_range:
        print(C)
        if kernel == 'linear':
            score, std, tscore, tstd = cv_loop_linear(l, X, cv, C=C,
                                                      n_repeats=n_repeats,
                                                      reduce_dim=reduce_dim)
        elif kernel == 'chi2':
            score, std, tscore, tstd = cv_loop_chi2(l, X, cv, C=C, n_repeats=n_repeats)
        elif kernel == 'rf':
            score, std, tscore, tstd = cv_loop_rf(l, X, cv, n_repeats=n_repeats)
            
        results['cv_C'][C] = {
            'score': score,
            'std': std,
            'tscore': tscore,
            'tstd': tstd
        }
        
    with open(resultsfile, 'w') as jf:
        json.dump(results, jf)


if __name__ == '__main__':
    svm_param_select()
