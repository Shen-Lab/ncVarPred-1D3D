#!/bin/sh
#GTEx eQTL

<<comment
for replicate in {1..5}
do
	for cellline in GM12878
	do
		for hic in ENCFF014VMM
		do
			#for seq in wt mt
			#do
			#	python danq_gcn_inference_gpu.py --our_model_name ${hic}_dnabert_replicate${replicate}.pkl --our_model_path ../model_assessment/concatenation_model/deepsea_bce_gcn_100000/ --sota_model_name DanQ_replicated.pkl --sota_model_path ../model_assessment/sota_model/ --seq_input_path GTEx_processed/ --seq_input_name GTEx_${cellline}_${seq}_seq.npy --structure_input_matching_name GTEx_${cellline}_structure_matching.npy --structure_input_path ../graph_data/if_matrix_100000/${hic}_novc_whole_normalized.npy --structure_input_matching_path GTEx_processed/ --feature_selected_boolean_path ../3dstructure_sanitycheck_data/deepsea_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name danq_gcn_replicate${replicate} --output_sota_name danq --seq_specific_name ${seq} --node_feature_type dnabert
			#done
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_danq_gcn_replicate${replicate} --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_danq --output_path model_prediction/
		done
	done
	for cellline in IMR90;
	do
		for hic in ENCFF928NJV;
		do
			#for seq in wt mt
			#do
			#	python danq_gcn_inference_gpu.py --our_model_name ${hic}_dnabert_replicate${replicate}.pkl --our_model_path ../model_assessment/concatenation_model/deepsea_bce_gcn_100000/ --sota_model_name DanQ_replicated.pkl --sota_model_path ../model_assessment/sota_model/ --seq_input_path GTEx_processed/ --seq_input_name GTEx_${cellline}_${seq}_seq.npy --structure_input_matching_name GTEx_${cellline}_structure_matching.npy --structure_input_path ../graph_data/if_matrix_100000/${hic}_novc_whole_normalized.npy --structure_input_matching_path GTEx_processed/ --feature_selected_boolean_path ../3dstructure_sanitycheck_data/deepsea_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name danq_gcn_replicate${replicate} --output_sota_name danq --seq_specific_name ${seq} --node_feature_type dnabert
			#done
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_danq_gcn_replicate${replicate} --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_danq --output_path model_prediction/
		done
	done
done
comment

#GRASP snp
#<<comment

for replicate in {1..5}
do
	for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
	do
		for label in 31kbp_negative_SNP 6.3kbp_negative_SNP Random_negative_SNP 360bp_negative_SNP 710bp_negative_SNP eQTL
		do
			for fold in {0..9}
			do
				#for seq in wt mt
				#do
				#	if test ! -f model_prediction/GRASP_${label}_${fold}_${hic}_danq_gcn_dnabert_replicate${replicate}_${seq}_prediction.npy
				#	then
				#	echo model_prediction/GRASP_${label}_${fold}_${hic}_gcn_dnabert_replicate${replicate}_${seq}_prediction.npy
				#	cp template_gpu.slurm run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
				#	echo "python danq_gcn_inference_gpu.py --our_model_name ${hic}_dnabert_replicate${replicate}.pkl --our_model_path ../model_assessment/concatenation_model/danq_bce_gcn_100000/ --sota_model_name DanQ_replicated.pkl --sota_model_path ../model_assessment/sota_model/ --seq_input_path GRASP_processed/ --seq_input_name GRASP_${label}_${fold}_${seq}_seq.npy --structure_input_matching_name GRASP_${label}_${fold}_structure_matching.npy --structure_input_path ../graph_data/if_matrix_100000/${hic}_novc_whole_normalized.npy --structure_input_matching_path GRASP_processed/ --feature_selected_boolean_path ../3dstructure_sanitycheck_data/deepsea_data/all_feature_index.npy --experiment_name GRASP_${label}_${fold} --output_path model_prediction/ --output_our_model_name ${hic}_danq_gcn_dnabert_replicate${replicate} --output_sota_name danq --seq_specific_name ${seq} --node_feature_type dnabert" >> run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
				#	sbatch run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
				#	rm run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
				#	fi
				#done
				if test ! -f model_prediction/GRASP_${label}_${fold}_${hic}_danq_gcn_dnabert_replicate${replicate}_diff.npy
				then
				echo GRASP_${label}_${fold}_${hic}_danq_gcn_dnabert_replicate${replicate}_diff.npy
				python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GRASP_${label}_${fold}_${hic}_danq_gcn_dnabert_replicate${replicate} --output_path model_prediction/
				fi
				if test ! -f model_prediction/GRASP_${label}_${fold}_danq_diff.npy
				then
				python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GRASP_${label}_${fold}_danq --output_path model_prediction/
				fi
			done
		done
	done
done

<<comment
for replicate in {1..5}
do
	for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
	do
		for label in 31kbp_negative_SNP 6.3kbp_negative_SNP Random_negative_SNP 360bp_negative_SNP 710bp_negative_SNP GWAS_Catalog
		do
			for fold in {0..9}
			do
				for seq in wt mt
				do
					if test ! -f model_prediction/GWAS_${label}_${fold}_${hic}_replicate${replicate}_${seq}_prediction.npy
					then
					echo model_prediction/GWAS_${label}_${fold}_${hic}_replicate${replicate}_${seq}_prediction.npy
					cp template_gpu.slurm run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
					echo "python deepsea_fc_inference_gpu.py --our_model_name ${hic}_replicate${replicate}.pkl --our_model_path ../model_assessment/concatenation_model/deepsea_bce_100000/ --sota_model_name deepsea_published.pkl --sota_model_path ../model_assessment/sota_model/ --seq_input_path GWAS_processed/ --seq_input_name GWAS_${label}_${fold}_${seq}_seq.npy --structure_input_matching_name GWAS_${label}_${fold}_structure_matching.npy --structure_input_path ../graph_data/if_matrix_100000/${hic}_novc_whole_normalized.npy --structure_input_matching_path GWAS_processed/ --feature_selected_boolean_path ../3dstructure_sanitycheck_data/deepsea_data/all_feature_index.npy --experiment_name GWAS_${label}_${fold} --output_path model_prediction/ --output_our_model_name ${hic}_replicate${replicate} --output_sota_name sota --seq_specific_name ${seq}" >> run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
					sbatch run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
					rm run_${replicate}_${hic}_${label}_${fold}_${seq}.slurm
					fi
				done
				if test ! -f model_prediction/GWAS_${label}_${fold}_${hic}_replicate${replicate}_log_odds_fc.npy
				then
				python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GWAS_${label}_${fold}_${hic}_replicate${replicate} --output_path model_prediction/
				fi
				if test ! -f model_prediction/GWAS_${label}_${fold}_sota_log_odds_fc.npy
				then
				python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GWAS_${label}_${fold}_sota --output_path model_prediction/
				fi
			done
		done
	done
done
comment


