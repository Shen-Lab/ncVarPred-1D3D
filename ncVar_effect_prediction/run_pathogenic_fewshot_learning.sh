#!/bin/sh

for lr in  1e-5
do
	for l2reg in 0
	do
		for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
		do
			for replicate in {1..5}
			do
				for fewshotsize in 50 100 150 200 250 300 350 400
				do
					mkdir -p model_replicate${replicate}
					cp template_gpu.slurm run.slurm
					echo "python cnn_mlp_diff_pathogenic_train.py --structure_name ${hic} --learning_rate ${lr} --retrain_boolean False --model_input_path na --model_output_path model_replicate${replicate}/ --epigenetic_embedding_model_input_path ../trained_model/CNN_MLP/${hic}_replicate${replicate}.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
					echo "python cnn_rnn_gcn_diff_pathogenic_train.py --structure_name ${hic} --learning_rate ${lr} --retrain_boolean False --model_input_path na --model_output_path model_replicate${replicate}/ --epigenetic_embedding_model_input_path ../trained_model/CNN_RNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
					sbatch run.slurm
					rm run.slurm
				done
			done
		done
	done
done

for lr in 1e-5
do
	for l2reg in 0
	do
		for fewshotsize in 50 100 150 200 250 300 350 400
		do
			cp template_gpu.slurm run.slurm
			echo "python deepsea_diff_pathogenic_train.py --learning_rate ${lr} --retrain_boolean False --model_input_path na --model_output_path model_replicate1/ --epigenetic_embedding_model_input_path ../trained_model/SOTA/DeepSEA_published.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
			echo "python danq_diff_pathogenic_train.py --learning_rate ${lr} --retrain_boolean False --model_input_path na --model_output_path model_replicate1/ --epigenetic_embedding_model_input_path ../trained_model/SOTA/DanQ_reproduced.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
			sbatch run.slurm
			rm run.slurm
		done
	done
done

