#!/bin/bash

# run final SVM 10x 10-fold cross-validation

# DATASET=data/full
DATASET=data/crop

echo "bow features"
for featurefile in ${DATASET}/features/*bow*.h5; do
    sbatch code/svm_param_select.py ${featurefile} --kernel chi2 -C 1 --n-per-class 200 --n-repeats 10 --seed 0;
done

echo
echo "vlad features"

for featurefile in ${DATASET}/features/*vlad*.h5; do
    sbatch code/svm_param_select.py ${featurefile} --kernel linear -C 1 --n-per-class 200 --n-repeats 10 --seed 0;
done

sbatch code/svm_param_select.py ${DATASET}/features/raw.h5 --kernel linear -C 1 --n-per-class 200 --n-repeats 10 --seed 0;

# scripts/svm_param_select.py data/full/features/vgg16_block4_conv3-vlad-32.h5 --kernel linear -C 1 --n-per-class 200 --n-repeats 10 --seed 0

# CNN features
# scripts/svm_param_select2.py data/afm3000/features/vgg16_block4_conv3-vlad-32.h5 --kernel linear -C 1 --n-per-class 200 --n-repeats 10 --seed 0
# scripts/svm_param_select2.py data/afm3000/features/vgg16_block4_conv3-vlad-100.h5 --kernel linear -C 1 --n-per-class 200 --n-repeats 10 --seed 0



# SIFT features
# scripts/svm_param_select2.py data/afm3000/features/ssift-vlad-100.h5 --kernel linear -C 1 --n-per-class 200 --n-repeats 10 --seed 0
