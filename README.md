# AFM CNN

## workflow

### Set up environment

* First, you will have to install CUDA 7 from NVidia, which I think requires setting up a developer account. Your machine might already have this, though.

`conda env create -f tf_env.yml`

This code has a LOT of dependencies, and the tf_env.yml file lists all of them so that Anaconda can install them. It will not automatically set up tensorflow, though... if things aren't working, try these commands in this order, after making your environment:

`conda install cudnn`

`conda install tensorflow-gpu`

`conda install keras-gpu`


### Download and unzip data

In command line, run:

`bash download_afm.sh`

But first:
* Change DATADIR to the folder you want to use to store data.
* This will download ~4000 image files and a single .csv file with the class labels and other metadata


### Perform feature mapping, classification, dimensionality reduction

Run:

`bash run_full_stack.sh`

But first:
* Again, change DATADIR to the same folder it was in the previous script
* Choose a value for TASK: currently 'fiber' and 'noise' are accepted values, and will classify fiber vs. not fiber or noisy vs. not noisy

Results are stored in various subfolders of DATADIR
