# AFM CNN

## workflow

1: unpack dataset (`microstructures.sqlite`, `micrographs/*.{tif,png,jpg}`, etc)
```sh
# bash download.sh
# get data from NIST for this project
# http://hdl.handle.net/11256/940
NIST_DATASET=11256/940
NIST_DATASET_URL=https://materialsdata.nist.gov/dspace/xmlui/bitstream/handle/${NIST_DATASET}

DATADIR=data

echo "download data files into DATADIR=${DATADIR}"

# download metadata
curl ${NIST_DATASET_URL}/microstructures.sqlite -o ${DATADIR}/microstructures.sqlite

# download micrographs
curl ${NIST_DATASET_URL}/micrographs.zip -o ${DATADIR}/micrographs.zip
unzip ${DATADIR}/micrographs.zip -d ${DATADIR}

```

2: pre-crop all the micrographs:
```sh
python scripts/crop_micrographs.py
```
3: generate json files mapping image keys to file paths (for inertia reasons...)
```sh
python scripts/enumerate_dataset.py
```

4: compute microstructure representations.
Representations are stored in hdf5 at `data/${format}/features/${representation}.h5`
```sh
bash scripts/compute_representations.sh
```

5: Run svm experiments. Cross-validation results stored in `data/${format}/svm/${representation}.json`
```sh
# primary microconstituent:
bash scripts/svm_result.sh

# annealing condition:
bash scripts/sample_svm.sh
```

6: run t-SNE (or other dimensionality reduction method...). Results stored in `data/${format}/tsne/${representation}.h5`
```sh
bash scripts/tsne_embed.sh
# bash scripts/manifold_embed.sh
```

7: make t-SNE thumbnail maps. Thumbnail maps are stored at `data/${format}/tsne/${representation}.png`
```sh
bash scripts/tsne_map.sh
```
