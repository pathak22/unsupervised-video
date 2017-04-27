#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
cd $DIR

FILE=torchmodels.tar.gz
URL=https://s3-us-west-1.amazonaws.com/unsupervised-video/$FILE
CHECKSUM=6ead77d7b387b51426ccc5d3c95f78bb

if [ ! -f $FILE ]; then
  echo "Downloading the unsupervised video torchmodels (803MB)..."
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
