#!/usr/bin/env python

""" controller script for uhcs project """
import os
import h5py
import json
import click
import numpy as np
import pandas as pd
from functools import partial
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans

# needed with slurm to see local python library under working dir
import sys
sys.path.append(os.getcwd())

from mfeat import io
from mfeat import cnn
from mfeat import encode

try:
    from mfeat import local # import this downstream in case octave and/or oct2py are not available
except OSError:
    print('make sure octave, vlfeat, and oct2py are installed to use SIFT features')
except ImportError:
    print('make sure octave, vlfeat, and oct2py are installed to use SIFT features')

layer_choices = cnn.layer_id.keys()

def ensure_dir(path):
    """ mkdir -p """
    try: os.makedirs(path)
    except: pass

@click.command()
@click.argument('micrographs_json', nargs=1, type=click.Path())
@click.option('-k', '--n_clusters', help='number of dictionary words', default=100)
@click.option('-s', '--style', default='ssift', type=click.Choice(['ssift', 'dsift', 'vgg16']),
              help='select image representation')
@click.option('-e', '--encoding', default='bow', type=click.Choice(['bow', 'vlad', 'fisher']),
              help='select image feature encoding method')
@click.option('-l', '--layername', default='block5_conv3', type=click.Choice(layer_choices),
              help='select vgg16 convolution layer')
@click.option('--multiscale/--no-multiscale', default=False,
              help='multiscale spatial pooling for CNN feature maps')
def featuremap2(micrographs_json, n_clusters, style, encoding, layername, multiscale):
    """ compute image representations for each image enumerated in micrographs_json 
        results are stored in HDF5 keyed by the image ids in micrographs_json
    """
    
    dataset_dir, __ = os.path.split(micrographs_json)

    ensure_dir(os.path.join(dataset_dir, 'dictionary'))
    ensure_dir(os.path.join(dataset_dir, 'features'))
                
    if style in ['vgg16']:
        if multiscale:
            method = '{}_multiscale_{}'.format(style, layername)
        else:
            method = '{}_{}'.format(style, layername)
    else:
        method = style
        
    metadata = {
        'dir': dataset_dir,
        'n_clusters': n_clusters,
        'method': method,
        'encoding': encoding
    }
    
    # obtain a dataset
    df_mg = pd.read_csv(micrographs_json)
    df_mg.imPath = [dataset_dir+'/'+str(i)+'.tif' for i in df_mg['id'].tolist()]

    keys = [str(i) for i in df_mg['id'].tolist()]
    micrographs = [io.load_image(file, barheight=0) for file in df_mg['imPath'].tolist()]

    # set up paths
    dictionary_file = '{dir}/dictionary/{method}-kmeans-{n_clusters}.pkl'.format(**metadata)
    featurefile = '{dir}/features/{method}-{encoding}-{n_clusters}.h5'.format(**metadata)

    if style == 'ssift':
        extract_func = lambda mic, fraction=1.0: local.sparse_sift(mic, fraction=fraction)
    elif style == 'dsift':
        extract_func = lambda mic, fraction=1.0: local.dense_sift(mic, fraction=fraction)
    elif style == 'vgg16':
        if multiscale:
            # use default scale parameters for now: 1/sqrt(2) with one octave of upsampling, 3 downsampling
            extract_func = lambda mic, fraction=1.0: cnn.multiscale_cnn_features(mic, layername, fraction=fraction)
        else:
            extract_func = lambda mic, fraction=1.0: cnn.cnn_features(mic, layername, fraction=fraction)

    try:
        dictionary = joblib.load(dictionary_file)
        print('loaded dictionary')
    except FileNotFoundError:
        print('learning dictionary for {} images'.format(len(micrographs)))
        training_fraction = 0.1

        results = map(partial(extract_func, fraction=training_fraction), micrographs)
        features = np.vstack(results)

        if encoding in ('bow', 'vlad'):
            dictionary = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=25)
        elif encoding in ('fisher'):
            dictionary = GMM(n_components=n_clusters, covariance_type='diag')

        print('clustering')
        dictionary.fit(features)
        print('done')
        # serialize k-means results to disk
        pkl_paths = joblib.dump(dictionary, dictionary_file)

    if encoding == 'bow':
        encode_func = lambda mic: encode.bag_of_words(extract_func(mic), dictionary)
    elif encoding == 'vlad':
        encode_func = lambda mic: encode.vlad(extract_func(mic), dictionary)
        
    features = map(encode_func, micrographs)
    with h5py.File(featurefile, 'w') as f:
        for key, feature in zip(keys, features):
            print(key)
            print(type(feature))
            f[key] = feature

    
    return

if __name__ == '__main__':
    featuremap2()
