#!/bin/sh

#  run_full_stack.sh
#  
#
#  Created by Nils Persson on 11/2/17.


# "Arguments" to this script are:
# the directory where the data is
# the name for the cleaned CSV database
# The classification task (column of CSV)

DATADIR=/data/nep1/afm
CLEANCSV=afm_clean.csv
TASK=noise


# Clean the main database:
# First argument is original afm.csv
# Second argument is output clean csv file (make sure it's the same folder as the original, can be named anything...

echo "cleaning database"
scripts/clean_afm_db.py ${DATADIR}/afm.csv ${DATADIR}/${CLEANCSV}


# Compute CNN representations
# feature map takes the database CSV as an input, and optional arguments:
# -s <vgg16, or other feature models>
# -e <vlad, or other feature embeddings>
# -k <int> size of embedding for VLAD
# -l <block4_conv3> layer of VGG16 to use

echo "computing representations"
#for v_size in 64 128 256; do
#    mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s vgg16 -e vlad -k ${v_size} -l block1_conv2
#done
#for v_size in 64 128 256; do
#    mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s vgg16 -e vlad -k ${v_size} -l block2_conv2
#done
#for v_size in 64 128 256; do
#    mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s vgg16 -e vlad -k ${v_size} -l block3_conv3
#done
#for v_size in 64 128 256; do
#    mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s vgg16 -e vlad -k ${v_size} -l block4_conv3
#done
#for v_size in 64 128 256; do
#    mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s vgg16 -e vlad -k ${v_size} -l block5_conv3
#done

#mfeat/bin/featuremap2.py ${DATADIR}/${CLEANCSV} -s cvsift -e vlad -k 512


# Train SVM for desired classification task

echo "training SVM"
for featurefile in ${DATADIR}/features/*vlad*.h5; do
scripts/svm_param_select2.py ${featurefile} ${DATADIR}/${CLEANCSV} ${TASK} --kernel linear -C 1 -n 50 -r 10;
done

# For training RandomForests, use kernel "rf", and C = n_estimators

#echo "training RandomForest"
#for featurefile in ${DATADIR}/features/*vlad*.h5; do
#scripts/svm_param_select2.py ${featurefile} ${DATADIR}/${CLEANCSV} ${TASK} --kernel rf -C 20 -r 1 -n 50;
#done


## t-SNE embedding
#
#echo "performing t-SNE embedding"
#for featurefile in ${DATADIR}/features/*vlad*.h5; do
#scripts/tsne_embed2.py ${featurefile} --kernel linear --n-repeats 10
#done
#
#
## t-SNE figure generation
#
#echo "generating t-SNE map"
#for featurefile in ${DATADIR}/tsne/*.h5; do
#scripts/tsne_map2.py ${featurefile} ${DATADIR}/${CLEANCSV} ${TASK} --perplexity 40 --bordersize 8
#done
#
