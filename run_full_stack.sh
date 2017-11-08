#!/bin/sh

#  run_full_stack.sh
#  
#
#  Created by Nils Persson on 11/2/17.


# "Arguments" to this script are:
# the directory where the data is
# the name for the cleaned CSV database
# The classification task (column of CSV)

DATADIR=data/afm3000
CLEANCSV=afm_clean.csv
TASK=fiber


# Clean the main database:
# First argument is original afm.csv
# Second argument is output clean csv file (make sure it's the same folder as the original, can be named anything...

#echo "cleaning database"
#scripts/clean_afm_db.py ${DATADIR}/afm.csv ${DATADIR}/${CLEANCSV}


# Compute CNN representations
# feature map takes the database CSV as an input, and optional arguments:
# -s <vgg16, or other feature models>
# -e <vlad, or other feature embeddings>
# -k <int> size of embedding for VLAD
# -l <block4_conv3> layer of VGG16 to use

#echo "computing representations"
#mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s vgg16 -e vlad -k 100 -l block4_conv3


# Train SVM for desired classification task

#echo "training SVM"
#for featurefile in ${DATADIR}/features/*vlad*.h5; do
#scripts/svm_param_select2.py ${featurefile} ${DATADIR}/${CLEANCSV} ${TASK} --kernel linear -C 1 -n 200 -r 10;
#done


# t-SNE embedding

echo "performing t-SNE embedding"
for featurefile in ${DATADIR}/features/*vlad*.h5; do
scripts/tsne_embed2.py ${featurefile} --kernel linear --n-repeats 10;
scripts/tsne_map2.py ${featurefile} ${DATADIR}/${CLEANCSV} ${TASK} --perplexity 40 --bordersize 8;
done


