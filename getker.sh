#!/bin/bash

out=$1
mr=$2
nr=$3
pa=$4
pb=$5
mri=$6

if [[
	( "$pa" != p && "$pa" != n ) ||
	( "$pb" != p && "$pb" != n ) || "$mri" == "" ]]; then

	echo "Usage: getker.sh <output_dir> <mr> <nr> <a_is_packed:p/n> <b_is_packed:p/n> <mr_inner>"
	echo "Get: output_dir=$out mr=$mr nr=$nr a_is_packed=$pa b_is_packed=$pb mr_inner=$mri"
	exit 1
fi

str="$mr $nr"

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

echo "$str" | exocc -o $out --stem ${mr}x${nr}_${pa}${pb} $(dirname $0)/getker.py

