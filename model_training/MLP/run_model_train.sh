#!/bin/sh

for lr in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
do
	#for resolution in 100000 500000 1000000;
	for resolution in 100000
	do
		for l1reg in 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12
		do
			for l2reg in 1e-6 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12
			do
				#for hic in ENCFF227XJZ ENCFF014VMM ENCFF563XES ENCFF482LGO ENCFF053BXY ENCFF812THZ ENCFF688KOY ENCFF777KBU ENCFF065LSP ENCFF718AWL ENCFF632MFV ENCFF355OWW ENCFF514XWQ ENCFF223UBX ENCFF799QGA ENCFF473CAA ENCFF999YXX ENCFF043EEE ENCFF029MPB ENCFF894GLR ENCFF997RGL ENCFF920CJR ENCFF928NJV ENCFF303PCK ENCFF366ERB ENCFF406HHC ENCFF013TGD ENCFF996XEO ENCFF464KRA ENCFF929RPW ENCFF097SKJ
				for hic in ENCFF014VMM ENCFF928NJV ENCFF013TGD
				do
					#To get the consistent results, we replicate the training process 5 times.
					for replicate in {1..5}
					do
						for margin in 1
						do
							#due to the submitted job time limits, you may want to retrain the model.
							mkdir -p model_bce_replicate${replicate}_round1
							cp template_gpu.slurm run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}_round1/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python danq_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round1/danq_bce_${hic}_resolution${resolution}_lr${lr}_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean True --retrain_boolean False --model_input_path na --model_output_path model_bce_replicate${replicate}_round1/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							#echo "python deepsea_train_gpu.py --structure_name ${hic} --resolution ${resolution} --learning_rate ${lr} --warmstart_boolean False --retrain_boolean True --model_input_path model_bce_replicate${replicate}_round1/deepsea_bce_${hic}_resolution${resolution}_lr${lr}_l1reg${l1reg}_l2reg${l2reg}_best.pkl --model_output_path model_bce_replicate${replicate}_round2/ --train_batch_size 512 --lambda_l1 ${l1reg} --lambda_l2 ${l2reg} --loss_type BCE" >> run_${hic}_${resolution}_${lr}.slurm
							sbatch run_${hic}_${resolution}_${lr}.slurm
							rm run_${hic}_${resolution}_${lr}.slurm
						done
					done
				done
			done
		done
	done
done

