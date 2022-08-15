#!/bin/sh

for pairwise_resolution in 100;
do
	mkdir -p data_correction_summary_${pairwise_resolution}/45degree_statistics_summary
	for seed in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900 7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700 9800 9900;
	do
		for hic in ENCFF014VMM;
		do
			if test ! -f data_correction_summary_${pairwise_resolution}/45degree_statistics_summary/${hic}_100_${seed}_similarity_summary.npy
			then
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step14_test_statistics_analysis_45degree.py --hic_name ${hic} --pairwise_resolution ${pairwise_resolution} --random_seed ${seed} --cellline_specific GM12878" >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
		for hic in ENCFF928NJV;
		do
			if test ! -f data_correction_summary_${pairwise_resolution}/45degree_statistics_summary/${hic}_100_${seed}_similarity_summary.npy
			then
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step14_test_statistics_analysis_45degree.py --hic_name ${hic} --pairwise_resolution ${pairwise_resolution} --random_seed ${seed} --cellline_specific IMR90" >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
		for hic in ENCFF013TGD;
		do
			if test ! -f data_correction_summary_${pairwise_resolution}/45degree_statistics_summary/${hic}_100_${seed}_similarity_summary.npy
			then
				cp template.slurm run_${hic}_${pairwise_resolution}_${seed}.slurm
				echo "python step14_test_statistics_analysis_45degree.py --hic_name ${hic} --pairwise_resolution ${pairwise_resolution} --random_seed ${seed} --cellline_specific K562" >> run_${hic}_${pairwise_resolution}_${seed}.slurm
				sbatch run_${hic}_${pairwise_resolution}_${seed}.slurm
				rm run_${hic}_${pairwise_resolution}_${seed}.slurm
			fi
		done
	done
done


