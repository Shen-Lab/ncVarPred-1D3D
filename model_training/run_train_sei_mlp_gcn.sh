#!/bin/sh

for lr in 1e-05
do
	for l2reg in 0
	do
		for replicate in {1..5}
		do
			for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
			do
				mkdir -p model_replicate${replicate}
				cp template_gpu.slurm run.slurm
    				#echo "python train_sei_mlp.py --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --structure_name ${hic} --model_input_path na --model_output_path model_replicate${replicate}/ --lambda_l2 ${l2reg}" >> run.slurm
                                #echo "python train_sei_gcn.py --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --structure_name ${hic} --model_input_path na --model_output_path model_replicate${replicate}/ --lambda_l2 ${l2reg} --node_feature_type DNABERT" >> run.slurm
				echo "python train_sei_gcn.py --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --structure_name ${hic} --model_input_path na --model_output_path model_replicate${replicate}/ --lambda_l2 ${l2reg} --node_feature_type allones" >> run.slurm
	   			sbatch run.slurm
				rm run.slurm
			done
		done
	done
done

