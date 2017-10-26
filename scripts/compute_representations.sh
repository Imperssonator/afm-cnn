#!/bin/bash

# dataset=data/full
dataset=data/crop

for dsize in 32 64 100; do
    j=$(sbatch mfeat/bin/featuremap.py ${dataset}/micrographs.json -s ssift -e bow -k ${dsize})
    jobid=$(echo ${j} | awk '{print $4}')
    sbatch --depend=afterok:${jobid} mfeat/bin/featuremap.py ${dataset}/micrographs.json -s ssift -e vlad -k ${dsize}
    
    j=$(sbatch mfeat/bin/featuremap.py ${dataset}/micrographs.json -s dsift -e bow -k ${dsize})
    jobid=$(echo ${j} | awk '{print $4}')
    sbatch --depend=afterok:${jobid} mfeat/bin/featuremap.py ${dataset}/micrographs.json -s dsift -e vlad -k ${dsize}
done

for conv in 4 5; do
    sbatch -n 6 mfeat/bin/featuremap.py ${dataset}/micrographs.json -s vgg16 -e vlad -k 32 -l block${conv}_conv3
done
	     

# mfeat/bin/featuremap.py data/full/micrographs.json -s vgg16 -e vlad -k 32 -l block4_conv3
# mfeat/bin/featuremap2.py data/afm/afm.csv -s vgg16 -e vlad -k 32 -l block4_conv3
# mfeat/bin/featuremap2.py data/afm3000/afm3000.csv -s ssift -e vlad -k 100
# mfeat/bin/featuremap2.py data/afm3000/afm3000.csv -s vgg16 -e vlad -k 100 -l block4_conv3
