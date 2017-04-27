#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
cd $DIR

FILE=caffemodels.tar.gz
URL=https://s3-us-west-1.amazonaws.com/unsupervised-video/$FILE
CHECKSUM=c4a5c022994e841e7e6225d842522f48

if [ ! -f $FILE ]; then
  echo "Downloading the unsupervised video caffemodels (829MB)..."
  wget $URL -O $FILE
  echo "Unzipping..."
  tar zxvf $FILE
  echo "Downloading Done."
else
  echo "File already exists. Checking md5..."
fi

os=`uname -s`
if [ "$os" = "Linux" ]; then
  checksum=`md5sum $FILE | awk '{ print $1 }'`
elif [ "$os" = "Darwin" ]; then
  checksum=`cat $FILE | md5`
elif [ "$os" = "SunOS" ]; then
  checksum=`digest -a md5 -v $FILE | awk '{ print $4 }'`
fi
if [ "$checksum" = "$CHECKSUM" ]; then
  echo "Checksum is correct. File was correctly downloaded."
  exit 0
else
  echo "Checksum is incorrect. DELETE and download again."
fi
