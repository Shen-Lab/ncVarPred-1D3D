#!/bin/sh

#GTEx eQTLs 
for cellline_i in GM12878 IMR90;
do
	python prepare_seq.py --seq_chr ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline_i}_eqtl_chr.npy --seq_coor ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline_i}_eqtl_coor.npy --seq_wt ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline_i}_eqtl_wt.npy --seq_mt ../ncVar_data/GTEx_eQTL_data/GTEx_${cellline_i}_eqtl_mt.npy --ref_seq_input_path ../ncVar_data/hg19_ref_seq/ --structure_matching_index_path ../training_data/if_matrix_matching_index_100000.txt --resolution 100000 --output_path ../ncVar_data/GTEx_eQTL_data/ --output_name GTEx_${cellline_i}
done


#ncVarDB
python prepare_seq.py --seq_chr ../ncVar_data/ncVarDB_data/ncvar_chr_list.npy --seq_coor ../ncVar_data/ncVarDB_data/ncvar_coor_list.npy --seq_wt ../ncVar_data/ncVarDB_data/ncvar_wt_list.npy --seq_mt ../ncVar_data/ncVarDB_data/ncvar_mt_list.npy --ref_seq_input_path ../ncVar_data/hg19_ref_seq/ --structure_matching_index_path ../training_data/if_matrix_matching_index_100000.txt --resolution 100000 --output_path ncVarDB/data/ --output_name ncvar


