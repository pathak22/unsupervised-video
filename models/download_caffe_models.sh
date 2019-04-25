#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
cd $DIR

FILE=caffemodels.tar.gz
URL=https://dl.fbaipublicfiles.com/unsupervised-video/$FILE
CHECKSUM=29e4a50f4fc77b0563a201f28577a895

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
