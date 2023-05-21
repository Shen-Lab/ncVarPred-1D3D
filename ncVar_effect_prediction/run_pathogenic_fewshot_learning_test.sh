#!/bin/sh

<<comment
for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
do
	for replicate in {1..5}
	do
		#for fewshotsize in 20 40 60 
		for fewshotsize in 80
		do
			cp template_gpu.slurm run.slurm
			echo "python cnn_mlp_diff_pathogenic_test.py --structure_name ${hic} --model_input_path ../trained_model/fewshot_pathogenic_model/CNN_related_pathogenic/CNN_MLP_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/CNN_MLP_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
			echo "python cnn_rnn_gcn_diff_pathogenic_test.py --structure_name ${hic} --model_input_path ../trained_model/fewshot_pathogenic_model/CNN_RNN_related_pathogenic/CNN_RNN_GCN_DNABERT_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/CNN_RNN_GCN_DNABERT_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
			sbatch run.slurm
			rm run.slurm
		done
	done
done
comment

<<comment
#for fewshotsize in 20 40 60 
for fewshotsize in 80
do
	cp template_gpu.slurm run.slurm
	echo "python deepsea_diff_pathogenic_test.py --model_input_path ../trained_model/fewshot_pathogenic_model/CNN_related_pathogenic/DeepSEA_diff_pathogenic_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/DeepSEA_diff_pathogenic_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
	echo "python danq_diff_pathogenic_test.py --model_input_path ../trained_model/fewshot_pathogenic_model/CNN_RNN_related_pathogenic/DanQ_diff_pathogenic_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/DeepSEA_diff_pathogenic_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
	sbatch run.slurm
	rm run.slurm	
done
comment


