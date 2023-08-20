#!/bin/bash

out=$1
mr=$2
nr=$3
pa=$4
pb=$5

if \
	[ $pa != p ] && [ $pa != n ] && \
	[ $pb != p ] && [ $pb != n ]; then
	echo "Usage: getker.sh <output_dir> <mr> <nr> <a_is_packed:p/n> <b_is_packed:p/n>"
	echo "Get: output_dir=$out mr=$mr nr=$nr a_is_packed=$pa b_is_packed=$pb"
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

echo "$str" | exocc -o $out --stem ${mr}x${nr}_${pa}${pb} $(dirname $0)/getker.py

