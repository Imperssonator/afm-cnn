#!/bin/sh

#  run_full_stack.sh
#  
#
#  Created by Nils Persson on 11/2/17.
#  

# Clean the main database:
# First argument is original afm.csv (may need to change path)
# Second argument is output clean csv file (make sure it's the same folder as the original, can be named anything... then pass the output csv as argument to featuremap2
scripts/clean_afm_db.py data/afm/afm.csv data/afm/afm_clean_test.csv

# feature map takes the database CSV as an input, and optional arguments:
# -s <vgg16, or other feature models>
# -e <vlad, or other feature embeddings>
# -k <int> size of embedding for VLAD
# -l <block4_conv3> layer of VGG16 to use

mfeat/bin/featuremap2.py data/afm3000/afm3000.csv -s vgg16 -e vlad -k 100 -l block4_conv3

