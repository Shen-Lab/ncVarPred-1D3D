#!/bin/sh

for pairwise_resolution in 500;
do
	mkdir -p data_correction_summary_${pairwise_resolution}
	for seed in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900;
	do
		for hic in ENCFF227XJZ ENCFF014VMM ENCFF563XES ENCFF482LGO ENCFF053BXY ENCFF812THZ ENCFF688KOY ENCFF777KBU ENCFF065LSP ENCFF718AWL ENCFF632MFV ENCFF355OWW ENCFF514XWQ ENCFF223UBX ENCFF799QGA ENCFF473CAA 
		do
			mkdir -p data_correction_summary_${pairwise_resolution}/pairwise_similarity_whole_${hic}_${pairwise_resolution}_${seed}
			if test ! -f data_correction_summary_${pairwise_resolution}/pairwise_similarity_whole_${hic}_${pairwise_resolution}_${seed}/${hic}_seq_similarity_output.npy;
			then
				echo ${hic}_${seed}
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step1_pairwise_similarity_analysis.py --hic_name ${hic} --cellline_name GM12878 --random_seed ${seed} --pairwise_resolution ${pairwise_resolution} " >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
		for hic in ENCFF999YXX ENCFF043EEE ENCFF029MPB ENCFF894GLR ENCFF997RGL ENCFF920CJR ENCFF928NJV ENCFF303PCK ENCFF366ERB;
		do
			mkdir -p data_correction_summary_${pairwise_resolution}/pairwise_similarity_whole_${hic}_${pairwise_resolution}_${seed}
			if test ! -f data_correction_summary_${pairwise_resolution}/pairwise_similarity_whole_${hic}_${pairwise_resolution}_${seed}/${hic}_seq_similarity_output.npy;
			then
				echo ${hic}_${seed}
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step1_pairwise_similarity_analysis.py --hic_name ${hic} --cellline_name IMR90 --random_seed ${seed} --pairwise_resolution ${pairwise_resolution} " >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
		for hic in ENCFF406HHC ENCFF013TGD ENCFF996XEO ENCFF464KRA ENCFF929RPW ENCFF097SKJ
		do
			mkdir -p data_correction_summary_${pairwise_resolution}/pairwise_similarity_whole_${hic}_${pairwise_resolution}_${seed}
			if test ! -f data_correction_summary_${pairwise_resolution}/pairwise_similarity_whole_${hic}_${pairwise_resolution}_${seed}/${hic}_seq_similarity_output.npy;
			then
				echo ${hic}_${seed}
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step1_pairwise_similarity_analysis.py --hic_name ${hic} --cellline_name K562 --random_seed ${seed} --pairwise_resolution ${pairwise_resolution} " >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
	done
done


