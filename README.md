# AFM CNN

## workflow

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
