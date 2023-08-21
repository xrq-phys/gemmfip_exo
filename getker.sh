#!/bin/bash

out=$1
dtype=$(echo $2 | awk '{ print tolower($0) }')
mr=$3
nr=$4
ta=$(echo $5 | awk '{ print tolower($0) }')
pa=$(echo $6 | awk '{ print tolower($0) }')
pb=$(echo $7 | awk '{ print tolower($0) }')
mri=$8

if [[
	( "$ta" != n && "$ta" != t ) ||
	( "$pa" != p && "$pa" != n ) ||
	( "$pb" != p && "$pb" != n ) || "$mri" == "" ]]; then

	echo "Usage: getker.sh <output_dir> <dtype> <mr> <nr> <a_transpose:n/t> <a_is_packed:p/n> <b_is_packed:p/n> <mr_inner>"
	echo "Get: output_dir=$out dtype=$dtype mr=$mr nr=$nr a_transpose=$ta a_is_packed=$pa b_is_packed=$pb mr_inner=$mri"
	exit 1
fi

str="$dtype $mr $nr"

if [ $ta = t ]; then
	str="$str 1"
else
	str="$str 0"
fi

if [ $pa = p ]; then
	str="$str 1"
else
	str="$str 0"
fi

if [ $pb = p ]; then
	str="$str 1"
else
	str="$str 0"
fi

str="$str $mri"

export PYTHONPATH=$(dirname $0)
echo "$str" | exocc -o $out --stem _${mr}x${nr}_${ta}n_${pa}${pb} $(dirname $0)/getker.py

