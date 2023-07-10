#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/generate_frostt_formats_onyx.sh <tensor_name.txt>

FORMATS=(
  sss012
)

BENCHMARKS=(
tensor3_elemadd
#tensor3_elemmul
tensor3_innerprod
tensor3_mttkrp
tensor3_ttm
tensor3_ttv
)

#export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
#export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt/
#export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
#export FROSTT_FORMATTED_TACO_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted/taco-tensor
#export FROSTT_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted

basedir=$(pwd)

for i in ${!FORMATS[@]}; do
    format=${FORMATS[@]};
    echo "Generating files for format $format..."
    
    $basedir/compiler/taco/build/bin/taco-test sam.pack_$format
    $basedir/compiler/taco/build/bin/taco-test sam.pack_other_frostt
    for b in ${!BENCHMARKS[@]}; do
	    bench=${BENCHMARKS[$b]}
    	while read line; do
        
        	name=$line 
        	echo "Generating input format files for $name..."
        	python $basedir/scripts/datastructure_tns.py -n $name -f $format -b $bench -hw
        	python $basedir/scripts/datastructure_tns.py -n $name -f $format --other -b $bench -hw
        	chmod -R 775 $FROSTT_FORMATTED_PATH
    	done <$1
    done
done
