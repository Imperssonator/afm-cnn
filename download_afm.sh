#!/bin/bash
# usage: `bash download.sh`

# get data from GA Tech Matin for this project
# https://matin.gatech.edu/resources/290/download/AFM4005.zip

MATIN_URL = https://matin.gatech.edu/resources/290/download/AFM4005.zip
CSV_URL=https://matin.gatech.edu/resources/298/download/afm.csv
DATADIR=/data/nep1

echo "download data files into DATADIR=${DATADIR}"

# download micrographs
curl ${MATIN_URL} -o ${DATADIR}/AFM.zip
unzip ${DATADIR}/AFM.zip -d ${DATADIR}/afm

curl ${CSV_URL} -o ${DATADIR}/afm/afm.csv
