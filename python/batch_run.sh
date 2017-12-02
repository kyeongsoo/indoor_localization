#!/bin/bash
# ns=( 3, 4, 5, 6, 7, 8, 9, 10 )
# ss=( 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 )
ns=( 1, 2 )
ss=( 0.0, 0.1 )
for (( n=1; n<=2; n++ ))
do
	for (( s=0; s<=1; s++ ))
	do
		scale=$(( s * 0.1 ))
		echo $n $scale
		# ./scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N $n --scaling $s
	done
done
# # N=3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 3 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 3 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 3 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 3 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 3 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 3 --scaling 0.5
# # N=4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 4 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 4 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 4 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 4 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 4 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 4 --scaling 0.5
# # N=5
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 5 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 5 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 5 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 5 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 5 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 5 --scaling 0.5
# # N=6
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 6 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 6 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 6 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 6 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 6 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 6 --scaling 0.5
# # N=7
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 7 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 7 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 7 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 7 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 7 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 7 --scaling 0.5
# # N=8
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 8 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 8 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 8 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 8 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 8 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 8 --scaling 0.5
# # N=9
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 9 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 9 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 9 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 9 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 9 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 9 --scaling 0.5
# # N=10
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 10 --scaling 0.0
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 10 --scaling 0.1
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 10 --scaling 0.2
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 10 --scaling 0.3
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 10 --scaling 0.4
# python scalable_indoor_localization.py -S "256,128,256" -C "64,128" -D 0.2 -N 10 --scaling 0.5
