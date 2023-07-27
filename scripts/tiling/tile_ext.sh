#!/bin/bash

# THIS FILE MUST BE RUN FROM sam/ location
# ./scripts/tiling/tile_ext.sh <tile_dir> <arch_config.yaml>

BENCHMARKS=(
#   matmul_ikj
	mat_mattransmul
)

sspath=$SUITESPARSE_PATH

basedir=$(pwd)

tiles_path=$basedir/extensor_mtx/$1

echo "$tiles_path"

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	rm -rf $basedir/tiles/*

	echo "Tiling mtx file"
	# python $basedir/sam/sim/src/tiling/tile.py --extensor --input_path $tiles_path --cotile $bench --multilevel --hw_config $basedir/sam/sim/src/tiling/$2 
	python3 ./sam/sim/src/tiling/tile.py --tensor_type ss --input_tensor rel5 --cotile mat_mattransmul --multilevel --hw_config ./sam/sim/src/tiling/memory_config_onyx.yaml --higher_order

	echo "Generating input format files for $tiles_path..."
	python3 $basedir/scripts/formatting/datastructure_suitesparse.py -n temp -hw -b $bench --input $basedir/tiles/$bench/mtx/ --output_dir_path $basedir/tiles/$bench/formatted --tiles

done