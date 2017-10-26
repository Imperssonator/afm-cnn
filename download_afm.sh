#!/bin/bash
# usage: `bash download.sh`

# get data from GA Tech Matin for this project
# https://matin.gatech.edu/resources/290/download/AFM4005.zip

MATIN_URL = https://matin.gatech.edu/resources/290/download/AFM4005.zip

DATADIR=/data/nep1

echo "download data files into DATADIR=${DATADIR}"

# download micrographs
curl ${MATIN_URL} -o ${DATADIR}/AFM.zip
unzip ${DATADIR}/AFM.zip -d ${DATADIR}/afm
