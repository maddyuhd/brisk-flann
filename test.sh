#!/bin/sh

for i in 4 8 16 32
do
	for j in 16 150 500
	do
		echo "-------------------------------------"
		echo B- $i, L- $j, Lmax- 100
		#r=$(($j+($j/5)))
		python main.py -n $i -l $j
		result=$(python search.py -n $i -i batch) #j $r)
		echo accuracy- "$result" %
	done
done

