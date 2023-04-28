#!/bin/sh

for pairwise_resolution in 50
do
	mkdir -p data_correction_summary_${pairwise_resolution}/45degree_statistics_summary
	for seed in 0 100 200
	do
		for hic in ENCFF227XJZ ENCFF014VMM ENCFF563XES ENCFF482LGO ENCFF053BXY ENCFF812THZ ENCFF688KOY ENCFF777KBU ENCFF065LSP ENCFF718AWL ENCFF632MFV ENCFF355OWW ENCFF514XWQ ENCFF223UBX ENCFF799QGA ENCFF473CAA ENCFF999YXX ENCFF043EEE ENCFF029MPB ENCFF894GLR ENCFF997RGL ENCFF920CJR ENCFF928NJV ENCFF303PCK ENCFF366ERB ENCFF406HHC ENCFF013TGD ENCFF996XEO ENCFF464KRA ENCFF929RPW ENCFF097SKJ
		do
			if test ! -f data_correction_summary_${pairwise_resolution}/45degree_statistics_summary/${hic}_${pairwise_resolution}_${seed}_count.npy;
			then
				echo ${hic}_${seed}
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step2_statistics_analysis_45degree.py --hic_name ${hic} --pairwise_resolution ${pairwise_resolution} --random_seed ${seed}" >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
	done
done


