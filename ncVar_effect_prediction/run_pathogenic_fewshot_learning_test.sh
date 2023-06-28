#!/bin/sh

#<<comment
for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
do
	for replicate in {1..5}
	do
		for fewshotsize in 50 `100 150 200 250 300 350 400
		do
			cp template_gpu.slurm run.slurm
			#echo "python cnn_mlp_diff_pathogenic_test.py --structure_name ${hic} --model_input_path model_selected_round6/CNN_MLP_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/CNN_MLP_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
			#echo "python cnn_rnn_gcn_diff_pathogenic_test.py --structure_name ${hic} --model_input_path model_selected_round6/CNN_RNN_GCN_DNABERT_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/CNN_RNN_GCN_DNABERT_diff_pathogenic_${hic}_replicate${replicate}_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
			#echo "python cnn_rnn_gcn_diff_pathogenic_test.py --structure_name ${hic} --model_input_path model_selected_round1/${hic}_replicate${replicate}.pkl --result_output_path model_prediction_round1/${hic}_replicate${replicate}_auroc_auprc.npy" >> run.slurm
			echo "python cnn_rnn_gcn_diff_pathogenic_test.py --structure_name ${hic} --model_input_path model_selected_cadd/CNN_RNN_GCN_DNABERT_diff_pathogenic_cadd_${hic}_replicate${replicate}.pkl --result_output_path model_prediction/CNN_RNN_GCN_DNABERT_diff_pathogenic_cadd_${hic}_replicate${replicate}_auroc_auprc.npy" >> run.slurm
			sbatch run.slurm
			rm run.slurm
		done
	done
done
#comment

<<comment
for fewshotsize in 50 100 150 200 250 300 350 400
do
	cp template_gpu.slurm run.slurm
	echo "python deepsea_diff_pathogenic_test.py --model_input_path model_selected_round6/DeepSEA_diff_pathogenic_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/DeepSEA_diff_pathogenic_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
	echo "python danq_diff_pathogenic_test.py --model_input_path model_selected_round6/DanQ_diff_pathogenic_fewshotsize${fewshotsize}.pkl --result_output_path model_prediction/DanQ_diff_pathogenic_fewshotsize${fewshotsize}_auroc_auprc.npy" >> run.slurm
	sbatch run.slurm
	rm run.slurm	
done
comment

