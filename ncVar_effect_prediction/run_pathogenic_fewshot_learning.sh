#!/bin/sh

#<<comment
for lr in  5e-5 5e-6
do
	for l2reg in 1e-8 1e-10 1e-12 0
	do
		for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
		do
			for replicate in {1..5}
			do
				for fewshotsize in 50
				# 100 150 200 250 300 350 400
				do
					mkdir -p model_replicate${replicate}
					cp template_gpu.slurm run.slurm
					#echo "python cnn_mlp_diff_pathogenic_train.py --structure_name ${hic} --learning_rate ${lr} --retrain_boolean True --model_input_path model_selected_round4/CNN_MLP_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}.pkl --model_output_path model_replicate${replicate}/ --epigenetic_embedding_model_input_path ../trained_model/CNN_MLP/${hic}_replicate${replicate}.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
					#echo "python cnn_rnn_gcn_diff_pathogenic_train.py --structure_name ${hic} --learning_rate ${lr} --retrain_boolean True --model_input_path model_selected_round4/CNN_RNN_GCN_DNABERT_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}.pkl --model_output_path model_replicate${replicate}/ --epigenetic_embedding_model_input_path ../trained_model/CNN_RNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
					echo "python cnn_rnn_gcn_diff_pathogenic_cadd_train.py --structure_name ${hic} --learning_rate ${lr} --retrain_boolean True --model_input_path model_selected_fewshotlearning_final/CNN_RNN_GCN_DNABERT_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize300.pkl --model_output_path model_replicate${replicate}/ --epigenetic_embedding_model_input_path ../trained_model/CNN_RNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --lambda_l2 ${l2reg}" >> run.slurm
					sbatch run.slurm
					rm run.slurm
				done
			done
		done
	done
done
#comment

<<comment
for lr in 2e-5
do
	for l2reg in 1e-8
	do
		for fewshotsize in 50 100 150 200 250 300 350 400
		do
			cp template_gpu.slurm run.slurm
			echo "python deepsea_diff_pathogenic_train.py --learning_rate ${lr} --retrain_boolean True --model_input_path model_selected_round5/DeepSEA_diff_pathogenic_fewshotsize${fewshotsize}.pkl --model_output_path model_replicate1/ --epigenetic_embedding_model_input_path ../trained_model/SOTA/DeepSEA_published.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
			echo "python danq_diff_pathogenic_train.py --learning_rate ${lr} --retrain_boolean True --model_input_path model_selected_round4/DanQ_diff_pathogenic_fewshotsize${fewshotsize}.pkl --model_output_path model_replicate1/ --epigenetic_embedding_model_input_path ../trained_model/SOTA/DanQ_reproduced.pkl --few_shot_size ${fewshotsize} --lambda_l2 ${l2reg}" >> run.slurm
			sbatch run.slurm
			rm run.slurm
		done
	done
done
comment

