#!/bin/sh

for hic in ENCFF014VMM
# ENCFF928NJV ENCFF013TGD
do
	cp template_gpu.slurm run.slurm
	#echo "python test_sei_mlp.py --model_path ../trained_model/Sei_MLP/Sei_${hic}.pkl --seq_label_path ../training_data/sei_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/sei_structure_matching_index/ --output_path model_prediction/ --output_model_name Sei_MLP_${hic} --model_version Sei_MLP" >> run.slurm
	echo "python test_sei_mlp.py --model_path ../trained_model/SOTA/Sei_reproduced.pkl --seq_label_path ../training_data/sei_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/sei_structure_matching_index/ --output_path model_prediction/ --output_model_name Sei --model_version Sei" >> run.slurm
	#echo "python test_sei_gcn.py --model_path ../trained_model/Sei_GCN/Sei_${hic}.pkl --seq_label_path ../training_data/sei_seq_label/ --structure_input_path ../training_data/if_matrix/${hic}_novc_whole_normalized.npy --structure_input_matching_path ../training_data/sei_structure_matching_index/ --output_path model_prediction/ --output_model_name Sei_GCN_${hic}" >> run.slurm
	sbatch run.slurm
	rm run.slurm
done
