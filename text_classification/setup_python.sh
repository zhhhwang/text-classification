#!/usr/bin/env bash

# For local running, this part can be ignore (Getting the data from s3)
S3_DATA_LOCATION=""
S3_GLOVE_DATA_LOCATION=""
aws s3 cp $S3_DATA_LOCATION data/interim
aws s3 cp $S3_GLOVE_DATA_LOCATION data/external

head -100000 data/interim/Reviews.csv > data/interim/Reviews_small.csv
unzip data/external/glove -d data/external/

# For setting up python
pip3 install -r requirements.txt
python3 -m nltk.downloader all