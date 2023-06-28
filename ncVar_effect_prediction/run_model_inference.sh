#!/bin/sh
mkdir -p model_prediction
#GTEx eQTL
<<comment
for replicate in {1..5}
do
	for cellline in GM12878
	do
		for hic in ENCFF014VMM
		do
			for seq in wt mt
			do
				python model_mlp_inference.py --our_model_path ../trained_model/CNN_MLP/${hic}_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_MLP_replicate${replicate} --output_sota_name DeepSEA --seq_specific_name ${seq}
				python model_mlp_inference.py --our_model_path ../trained_model/CNN_RNN_MLP/${hic}_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_RNN_MLP_replicate${replicate} --output_sota_name DanQ --seq_specific_name ${seq}
				python model_gcn_inference.py --our_model_path ../trained_model/CNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_GCN_DNABERT_replicate${replicate} --output_sota_name DeepSEA --seq_specific_name ${seq}
				python model_gcn_inference.py --our_model_path ../trained_model/CNN_RNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_RNN_GCN_DNABERT_replicate${replicate} --output_sota_name DanQ --seq_specific_name ${seq}
			done
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_MLP_replicate${replicate} --output_path model_prediction/
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_RNN_MLP_replicate${replicate} --output_path model_prediction/
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_GCN_DNABERT_replicate${replicate} --output_path model_prediction/
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_RNN_GCN_DNABERT_replicate${replicate} --output_path model_prediction/
		done
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_DeepSEA --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_DanQ --output_path model_prediction/
	done
	for cellline in IMR90;
	do
		for hic in ENCFF928NJV;
		do
			for seq in wt mt
			do
				python model_mlp_inference.py --our_model_path ../trained_model/CNN_MLP/${hic}_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_MLP_replicate${replicate} --output_sota_name DeepSEA --seq_specific_name ${seq}
				python model_mlp_inference.py --our_model_path ../trained_model/CNN_RNN_MLP/${hic}_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_RNN_MLP_replicate${replicate} --output_sota_name DanQ --seq_specific_name ${seq}
				python model_gcn_inference.py --our_model_path ../trained_model/CNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_GCN_DNABERT_replicate${replicate} --output_sota_name DeepSEA --seq_specific_name ${seq}
				python model_gcn_inference.py --our_model_path ../trained_model/CNN_RNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_${seq}_seq.npy --structure_matching_path ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline}_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/GTEx_eQTL_data/${cellline}_feature_index.npy --experiment_name GTEx_${cellline} --output_path model_prediction/ --output_our_model_name CNN_RNN_GCN_DNABERT_replicate${replicate} --output_sota_name DanQ --seq_specific_name ${seq}
			done
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_MLP_replicate${replicate} --output_path model_prediction/
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_RNN_MLP_replicate${replicate} --output_path model_prediction/
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_GCN_DNABERT_replicate${replicate} --output_path model_prediction/
			python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_CNN_RNN_GCN_DNABERT_replicate${replicate} --output_path model_prediction/
		done
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_DeepSEA --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name GTEx_${cellline}_DanQ --output_path model_prediction/
	done
done
comment

#ncVarDB
#<<comment
for replicate in {1..5}
do
	for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
	do
		for seq in wt mt
		do
			python model_mlp_inference.py --our_model_path ../trained_model/CNN_MLP/${hic}_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_path ../ncVar_data/ncVarDB_data/ncvar_${seq}_seq.npy --structure_matching_path ../ncVar_data/ncVarDB_data/ncvar_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/ncVarDB_data/all_feature_index.npy --experiment_name ncVar --output_path model_prediction/ --output_our_model_name CNN_MLP_${hic}_replicate${replicate} --output_sota_name DeepSEA --seq_specific_name ${seq}
			python model_mlp_inference.py --our_model_path ../trained_model/CNN_RNN_MLP/${hic}_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_path ../ncVar_data/ncVarDB_data/ncvar_${seq}_seq.npy --structure_matching_path ../ncVar_data/ncVarDB_data/ncvar_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/ncVarDB_data/all_feature_index.npy --experiment_name ncVar --output_path model_prediction/ --output_our_model_name CNN_RNN_MLP_${hic}_replicate${replicate} --output_sota_name DanQ --seq_specific_name ${seq}
			python model_gcn_inference.py --our_model_path ../trained_model/CNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_path ../ncVar_data/ncVarDB_data/ncvar_${seq}_seq.npy --structure_matching_path ../ncVar_data/ncVarDB_data/ncvar_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/ncVarDB_data/all_feature_index.npy --experiment_name ncVar --output_path model_prediction/ --output_our_model_name CNN_GCN_${hic}_DNABERT_replicate${replicate} --output_sota_name DeepSEA --seq_specific_name ${seq}
			python model_gcn_inference.py --our_model_path ../trained_model/CNN_RNN_GCN/${hic}_DNABERT_replicate${replicate}.pkl --sota_model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_path ../ncVar_data/ncVarDB_data/ncvar_${seq}_seq.npy --structure_matching_path ../ncVar_data/ncVarDB_data/ncvar_structure_matching.npy --structure_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --feature_selected_boolean_path ../ncVar_data/ncVarDB_data/all_feature_index.npy --experiment_name ncVar --output_path model_prediction/ --output_our_model_name CNN_RNN_GCN_${hic}_DNABERT_replicate${replicate} --output_sota_name DanQ --seq_specific_name ${seq}
		done
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name ncVar_CNN_MLP_${hic}_replicate${replicate} --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name ncVar_CNN_RNN_MLP_${hic}_replicate${replicate} --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name ncVar_CNN_GCN_${hic}_DNABERT_replicate${replicate} --output_path model_prediction/
		python get_log_odds_fc.py --input_path model_prediction/ --experiment_name ncVar_CNN_RNN_GCN_${hic}_DNABERT_replicate${replicate} --output_path model_prediction/
	done
done
python get_log_odds_fc.py --input_path model_prediction/ --experiment_name ncVar_DeepSEA --output_path model_prediction/
python get_log_odds_fc.py --input_path model_prediction/ --experiment_name ncVar_DanQ --output_path model_prediction/
#comment

