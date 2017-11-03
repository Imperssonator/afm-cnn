#!/usr/bin/env python

import os
import click
import pandas as pd
import numpy as np

@click.command()
@click.argument('orig_csv', type=click.Path())
@click.argument('out_csv', type=click.Path())
def clean_afm_db(orig_csv, out_csv):
    
    df_mg = pd.read_csv(orig_csv)  # open csv to dataframe
    df_mg = df_mg.loc[0:3360]  # take rows that have already been labeled
    df_mg.imPath = [os.path.split(orig_csv)[0]+'/'+str(i)+'.tif' for i in df_mg['id'].tolist()]  # build file names from ids
    
    df_mg = df_mg[df_mg.noise!='x']  # Remove bad images
    df_mg = df_mg[df_mg.channel!='AmplitudeActual']  # Remove channels that are low population
    df_mg = df_mg[df_mg.channel!='DeflectionActual']
    
    # Fill NaNs
    df_mg['noise'] = df_mg['noise'].fillna(value='c')  # c for 'clean'
    df_mg['fiber'] = df_mg['fiber'].fillna(value='n')  # n for 'not fiber'
    df_mg['channel'] = df_mg['channel'].str.replace('ZSensor','Height')  # these channels are essentially equivalent

    # Simplify the noise category to 'noise' or 'clean'
    df_mg.noise[df_mg['noise']!='c']='n'

    # Write out the file
    df_mg.to_csv(path_or_buf=out_csv, index=False)

if __name__ == '__main__':
    clean_afm_db()