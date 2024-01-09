#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/generate_frostt_formats_onyx.sh <tensor_name.txt>

FORMATS=(
  sss012
)

BENCHMARKS=(
  #using all tensor apps except elemmul here**
  # tensor3_elemadd
  # tensor3_innerprod
  tensor3_ttv
  # tensor3_elemmul
  # tensor3_mttkrp
  # tensor3_ttm
  # using tensor3_ttm
)

# OTHERBENCHES='["tensor3_ttv"]'
# export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
# export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt/
# export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
# export FROSTT_FORMATTED_TACO_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted/taco-tensor
# export FROSTT_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted

export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
export FROSTT_PATH=/home/avb03/sparse-datasets/tensors
export SUITESPARSE_FORMATTED_PATH=/home/avb03/sam/SUITESPARSE_FORMATTED
export FROSTT_FORMATTED_TACO_PATH=/home/avb03/sam/FROST_FORMATTED_TACO
export FROSTT_FORMATTED_PATH=/home/avb03/sam/FROST_FORMATTED
export TACO_TENSOR_PATH=/home/avb03/sam/TACO_TENSOR

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
        	python3 $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format -b $bench -hw
        	python3 $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format --other -b $bench -hw
        	# if [[ $OTHERBENCHES =~ "$bench" ]]; then
			    #   echo "Generating format of 'other' tensor"
			    #   python3 $basedir/scripts/datastructure_tns_old.py -n $line -f ss01 --other -ss -b $bench -hw
		      # fi
          chmod -R 775 $FROSTT_FORMATTED_PATH
    	done <$1
    done
done