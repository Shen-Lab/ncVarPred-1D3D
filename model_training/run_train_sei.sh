#!/bin/sh

for lr in 1e-5
do
	for replicate in 1
	do
		for l2reg in 0
		do
			mkdir -p model_replicate${replicate}
			cp template_gpu.slurm run.slurm
			echo "python train_sei.py --learning_rate ${lr} --retrain_boolean False --model_input_path na --l2reg ${l2reg} --model_output_path model_replicate${replicate}/" >> run.slurm
			sbatch run.slurm
			rm run.slurm
		done
	done
done

