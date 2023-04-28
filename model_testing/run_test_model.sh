for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
do
	for replicate in {1..5}
	do
		mkdir -p model_prediction
		cp template_gpu.slurm run.slurm
		echo "python test_model_mlp.py --model_path ../trained_model/CNN_MLP/${hic}_replicate${replicate}.pkl --seq_label_path ../training_data/deepsea_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/deepsea_structure_matching_index/ --output_path model_prediction/ --output_model_name CNN_MLP_${hic}_replicate${replicate} --model_version CNN_MLP" >> run.slurm
		echo "python test_model_mlp.py --model_path ../trained_model/CNN_RNN_MLP/${hic}_replicate${replicate}.pkl --seq_label_path ../training_data/deepsea_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/deepsea_structure_matching_index/ --output_path model_prediction/ --output_model_name CNN_RNN_MLP_${hic}_replicate${replicate} --model_version CNN_RNN_MLP" >> run.slurm
		echo "python test_model_mlp.py --model_path ../trained_model/SOTA/DeepSEA_published.pkl --seq_label_path ../training_data/deepsea_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/deepsea_structure_matching_index/ --output_path model_prediction/ --output_model_name DeepSEA --model_version CNN" >> run.slurm
		echo "python test_model_mlp.py --model_path ../trained_model/SOTA/DanQ_reproduced.pkl --seq_label_path ../training_data/deepsea_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/deepsea_structure_matching_index/ --output_path model_prediction/ --output_model_name DanQ --model_version CNN_RNN" >> run.slurm
		for node_feature_type in DNABERT DeepSEA allones
		do
			echo "python test_model_gcn.py --model_path ../trained_model/CNN_GCN/${hic}_${node_feature_type}_replicate${replicate}.pkl --seq_label_path ../training_data/deepsea_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/deepsea_structure_matching_index/ --node_feature_type ${node_feature_type} --output_path model_prediction/ --output_model_name CNN_GCN_${hic}_${node_feature_type}_replicate${replicate}" >> run.slurm
		done
		for node_feature_type in DNABERT allones
		do
			echo "python test_model_gcn.py --model_path ../trained_model/CNN_RNN_GCN/${hic}_${node_feature_type}_replicate${replicate}.pkl --seq_label_path ../training_data/deepsea_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/deepsea_structure_matching_index/ --node_feature_type ${node_feature_type} --output_path model_prediction/ --output_model_name CNN_RNN_GCN_${hic}_${node_feature_type}_replicate${replicate}" >> run.slurm
		done
		sbatch run.slurm
		rm run.slurm
	done
done



