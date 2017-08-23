#!/bin/sh

for i in 2 4 8 16 32 64
do
	for j in 16 150 500
	do
		r=$(($j+($j/5)))
		python main.py $i $j
		result=$(python traverse.py $j $r)
		echo B- $i, L- $j, Lmax- $r, accuracy- "$result" %
		echo "-------------------------------------"
	done
done

