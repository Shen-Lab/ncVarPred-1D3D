#!/bin/sh

for lr in 5e-05
do
	for resolution in 100000;
	do
		for l1reg in 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12
		do
			for l2reg in 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12
			do
				#for hic in ENCFF227XJZ ENCFF014VMM ENCFF563XES ENCFF482LGO ENCFF053BXY ENCFF812THZ ENCFF688KOY ENCFF777KBU ENCFF065LSP ENCFF718AWL ENCFF632MFV ENCFF355OWW ENCFF514XWQ ENCFF223UBX ENCFF799QGA ENCFF473CAA ENCFF999YXX ENCFF043EEE ENCFF029MPB ENCFF894GLR ENCFF997RGL ENCFF920CJR ENCFF928NJV ENCFF303PCK ENCFF366ERB ENCFF406HHC ENCFF013TGD ENCFF996XEO ENCFF464KRA ENCFF929RPW ENCFF097SKJ
				for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
				do
					for replicate in {1..5}
					do
						for node_feature_type  in dnabert allones
						do
							mkdir -p model_bce_replicate${replicate}_round1
							cp template_gpu.slurm run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_gcn_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}_round1/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --node_feature_type ${node_feature_type} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_gcn_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round1/deepsea_gcn_${node_feature_type}_bce_${hic}_resolution${resolution}_lr${lr}_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --node_feature_type ${node_feature_type} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_gat_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}_round1/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --node_feature_type ${node_feature_type} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_gat_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round1/deepsea_gat_${node_feature_type}_bce_${hic}_resolution${resolution}_lr${lr}_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --node_feature_type ${node_feature_type} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_gcn_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}_round1/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --node_feature_type ${node_feature_type} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_gcn_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round1/danq_gcn_${node_feature_type}_bce_${hic}_resolution${resolution}_lr${lr}_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --node_feature_type ${node_feature_type} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							sbatch run_${hic}_${resolution}_${lr}.slurm
							rm run_${hic}_${resolution}_${lr}.slurm
						done
					done
				done
			done
		done
	done
done

