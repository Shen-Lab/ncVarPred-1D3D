#!/bin/sh

for lr in 5e-05
do
	#for resolution in 100000 500000 1000000;
	for resolution in 100000
	do
		for l1reg in 1e-11 1e-12
		do
			for l2reg in 1e-8 1e-9
			do
				#for hic in ENCFF227XJZ ENCFF014VMM ENCFF563XES ENCFF482LGO ENCFF053BXY ENCFF812THZ ENCFF688KOY ENCFF777KBU ENCFF065LSP ENCFF718AWL ENCFF632MFV ENCFF355OWW ENCFF514XWQ ENCFF223UBX ENCFF799QGA ENCFF473CAA ENCFF999YXX ENCFF043EEE ENCFF029MPB ENCFF894GLR ENCFF997RGL ENCFF920CJR ENCFF928NJV ENCFF303PCK ENCFF366ERB ENCFF406HHC ENCFF013TGD ENCFF996XEO ENCFF464KRA ENCFF929RPW ENCFF097SKJ
				for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
				do
					for replicate in {1..5}
					do
						for margin in 1
						do
							mkdir -p model_bce_replicate${replicate}_round1
							#mkdir -p model_ap_replicate${replicate}
							cp template_gpu.slurm run_${hic}_${resolution}_${lr}.slurm
							echo "python danq_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}_round1/ --train_batch_size 64 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type AP" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round1/danq_bce_${hic}_resolution${resolution}_lr0.0001_l1reg1e-11_l2reg1e-9_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_train_test_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round2/danq_bce_${hic}_resolution${resolution}_lr5e-05_l1reg1e-11_l2reg1e-9_best.pkl --model_output_path model_bce_replicate${replicate}_round3/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round3/danq_bce_${hic}_resolution${resolution}_lr5e-05_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round4/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}/ --train_batch_size 64 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}/deepsea_bce_${hic}_resolution${resolution}_lr${lr}_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 64 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_train_${resolution}_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_ap_replicate${replicate}/ --train_batch_size 32 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type AP --sh_margin ${margin}" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_train_${resolution}_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_ap_replicate${replicate}/deepsea_ap_margin1.0_${hic}_resolution${resolution}_lr1e-05_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_ap_replicate${replicate}_round2/ --train_batch_size 32 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type AP --sh_margin ${margin}" >> run_${hic}_${resolution}_${lr}.slurm
							sbatch run_${hic}_${resolution}_${lr}.slurm
							rm run_${hic}_${resolution}_${lr}.slurm
						done
					done
				done
			done
		done
	done
done

