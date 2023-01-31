#!/bin/bash

basedir=$(pwd)
path=$basedir/jsons
download_script=scripts/download_suitesparse_stream_overhead.sh

mkdir -p $path

bench=$2

# Create download_script that downloads ONLY the suitesparse matrices listed in the text file that is passed in as the first argument of this script
[ -e $download_script ] && rm $download_script
echo "mkdir -p ${SUITESPARSE_PATH}" >> $download_script
echo "pushd ." >> $download_script
echo "cd ${SUITESPARSE_PATH}" >> $download_script
grep -F -f $1 scripts/download_suitesparse.sh >> $download_script 
echo "popd" >> $download_script

# Make it an executable
chmod ugo+x $download_script

# Call the download_script (created above)
./$download_script

# Unpack the downloaded suitesparse files since they come in .tar format
./scripts/unpack_suitesparse.sh $(realpath $1)

while read line; do
	echo "Generating input format files for $line..."
	sspath=${SUITESPARSE_PATH}/$line
	SUITESPARSE_TENSOR_PATH=$sspath python $basedir/scripts/datastructure_suitesparse.py -n $line -hw -b $bench 
done <$(realpath $1)
