#!/bin/sh

for lr in 1e-5
do
	for l1reg in 0
	do
		for l2reg in 0
		do
			for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
			do
				for replicate in {1..5}
				do
					mkdir -p model_replicate${replicate}
					cp template_gpu.slurm run_model_training.slurm
					echo "python train_model_mlp.py --structure_name ${hic} --learning_rate ${lr} --model_version CNN_MLP --warmstart_boolean True --warmstart_model_path ../trained_model/SOTA/DeepSEA_published.pkl --retrain_boolean False --model_input_path na --model_output_path model_replicate${replicate}/ --lambda_l1 ${l1reg} --lambda_l2 ${l2reg}" >> run_model_training.slurm
					echo "python train_model_mlp.py --structure_name ${hic} --learning_rate ${lr} --model_version CNN_RNN_MLP --warmstart_boolean True --warmstart_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --retrain_boolean False --model_input_path na --model_output_path model_replicate${replicate}/ --lambda_l1 ${l1reg} --lambda_l2 ${l2reg}" >> run_model_training.slurm
					for node_feature_type in DNABERT DeepSEA allones
					do
						echo "python train_model_gcn.py --structure_name ${hic} --learning_rate ${lr} --model_version CNN_GCN --warmstart_boolean True --warmstart_model_path ../trained_model/SOTA/DeepSEA_published.pkl --retrain_boolean False --model_input_path na --model_output_path model_replicate${replicate}/ --node_feature_type ${node_feature_type} --lambda_l1 ${l1reg} --lambda_l2 ${l2reg}" >> run_model_training.slurm
						echo "python train_model_gcn.py --structure_name ${hic} --learning_rate ${lr} --model_version CNN_RNN_GCN --warmstart_boolean True --warmstart_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --retrain_boolean False --model_input_path na --model_output_path model_replicate${replicate}/ --node_feature_type ${node_feature_type} --lambda_l1 ${l1reg} --lambda_l2 ${l2reg}" >> run_model_training.slurm
					done
					sbatch run_model_training.slurm
					rm run_model_training.slurm
				done
			done
		done
	done
done

