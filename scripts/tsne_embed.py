#!/usr/bin/env python
import os
import h5py
import click
import numpy as np
import warnings
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel

import ntsne

# needed with slurm to see local python library under working dir
import sys
sys.path.append(os.path.join(os.getcwd(), 'code'))

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

def stash_tsne_embeddings(resultsfile, keys, embeddings, perplexity):

    with h5py.File(resultsfile) as f:
        g = f.create_group('perplexity-{}'.format(perplexity))
        for idx, key in enumerate(keys):
            # add t-SNE map point for each record
            g[key] = embeddings[idx]
        return


@click.command()
@click.argument('datafile', type=click.Path())
@click.option('--kernel', '-k', type=click.Choice(['linear', 'chi2']), default='linear')
@click.option('--n-repeats', '-r', type=int, default=1)
@click.option('--seed', '-s', type=int, default=None)
def tsne_embed(datafile, kernel, n_repeats, seed):
    # datafile = './data/full/features/vgg16_block5_conv3-vlad-32.h5'
    print('working')
    resultsfile = datafile.replace('features', 'tsne')
    
    keys, features = load_representations(datafile)
    
    # Use pandas here instead of wacky sqlite stuff
    ids = pd.Series([int(s) for s in keys])
    df_mg = pd.read_csv('/Users/Imperssonator/CC/uhcs/data/afm3000/afm3000.csv')
    which_ids = np.where(ids.apply(lambda x: x in df_mg.id.tolist()))[0]
    keys_reduced = keys[which_ids]
    ids_reduced = pd.Series([int(s) for s in keys_reduced])
    features_reduced = features[which_ids,:]
    df_mg = df_mg.set_index('id')
    labels = np.array(df_mg['fiber'].loc[ids_reduced.tolist()])
    
#    labels = []
#    for key in keys:
#        if '-' in key:
#            # deal with cropped micrographs: key -> Micrograph.id-UL
#            m_id, quadrant = key.split('-')
#        else:
#            m_id = key
#        m = db.query(Micrograph).filter(Micrograph.micrograph_id == int(m_id)).one()
#        labels.append(m.primary_microconstituent)
#    labels = np.array(labels)

    if kernel == 'linear':
        x_pca = PCA(n_components=50).fit_transform(features)
    elif kernel == 'chi2':
        gamma = -1 / np.mean(additive_chi2_kernel(features))

        with warnings.catch_warnings():
            warnings.simplefilter("once", DeprecationWarning)
            x_pca = KernelPCA(n_components=50, kernel=chi2_kernel, gamma=gamma).fit_transform(features)
        
    perplexity = [10, 20, 30, 40, 50, 60]
    for p in perplexity:
        x_tsne = ntsne.best_tsne(x_pca, perplexity=p, theta=0.1, n_repeats=n_repeats)
        stash_tsne_embeddings(resultsfile, keys, x_tsne, p)
        
if __name__ == '__main__':
    tsne_embed()
