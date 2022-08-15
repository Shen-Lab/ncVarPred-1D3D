#!/bin/sh

for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
do
	#for resolution in 100000 500000 1000000
	for resolution in 100000
	do
		for replicate in {1..5}
		do
			#for loss in bce
			#do
			#	cp template_gpu.slurm run_${hic}_${resolution}_${replicate}_${loss}.slurm
				#echo "python deepsea_inference_gpu.py --resolution ${resolution} --model_path concatenation_model/deepsea_bce_${resolution}/${hic}_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction_selected/ --output_model_name deepsea_${hic}_fc_${loss}_${resolution}_replicate${replicate} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}_${loss}.slurm
				#echo "python deepsea_inference_gpu.py --resolution ${resolution} --model_path concatenation_model/danq_bce_${resolution}/${hic}_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction/ --output_model_name danq_${hic}_fc_${loss}_${resolution}_replicate${replicate} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}_${loss}.slurm
				#echo "python deepsea_inference_gpu.py --resolution ${resolution} --model_path sota_model/deepsea_published.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction_selected/ --output_model_name deepsea_sota --architecture_name deepsea" >> run_${hic}_${resolution}_${replicate}_${loss}.slurm
				#echo "python deepsea_inference_gpu.py --resolution ${resolution} --model_path sota_model/DanQ_replicated.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction_selected/ --output_model_name danq_sota --architecture_name deepsea" >> run_${hic}_${resolution}_${replicate}_${loss}.slurm
				#sbatch run_${hic}_${resolution}_${replicate}_${loss}.slurm
				#rm run_${hic}_${resolution}_${replicate}_${loss}.slurm
			#done
			#for node_feature_type in dnabert allones
			for node_feature_type  in dnabert
			do
				cp template_gpu.slurm run_${hic}_${resolution}_${replicate}.slurm
				#for model in deepsea danq
				for model in danq
				do
					#echo "python deepsea_gcn_inference.py --model_path concatenation_model/${model}_bce_gcn_${resolution}/${hic}_${node_feature_type}_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction/ --output_model_name ${model}_${hic}_gcn_${node_feature_type}_replicate${replicate} --node_feature_type ${node_feature_type} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}.slurm
					#echo "python deepsea_gcn_binary_inference.py --model_path concatenation_model/${model}_bce_gcn_${resolution}/${hic}_${node_feature_type}_binary_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction/ --output_model_name ${model}_${hic}_gcn_${node_feature_type}_binary_replicate${replicate} --node_feature_type ${node_feature_type} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}.slurm
					#echo "python ${model}_gcn_extract_motif.py --model_path concatenation_model/${model}_bce_gcn_${resolution}/${hic}_${node_feature_type}_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction/ --output_model_name ${model}_${hic}_gcn_${node_feature_type}_replicate${replicate} --node_feature_type ${node_feature_type} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}.slurm
					echo "python ${model}_fc_extract_motif.py --resolution ${resolution} --model_path concatenation_model/${model}_bce_${resolution}/${hic}_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction/ --output_model_name ${model}_${hic}_fc_replicate${replicate} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}.slurm
				done
				#echo "python deepsea_gat_inference.py --model_path concatenation_model/deepsea_bce_gat_${resolution}/${hic}_${node_feature_type}_replicate${replicate}.pkl --seq_label_path ../graph_data/concatenation_input/end_to_end_concatenation_input_seq_label/ --structure_input_path ../graph_data/if_matrix_${resolution}/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../graph_data/concatenation_input/end_to_end_concatenation_input_${resolution}/ --output_path model_prediction/ --output_model_name deepsea_${hic}_gat_${node_feature_type}_replicate${replicate} --node_feature_type ${node_feature_type} --architecture_name deepsea_concatenation" >> run_${hic}_${resolution}_${replicate}.slurm
				sbatch run_${hic}_${resolution}_${replicate}.slurm
				rm run_${hic}_${resolution}_${replicate}.slurm
			done
		done
	done
done

