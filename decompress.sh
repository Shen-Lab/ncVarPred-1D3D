mkdir -p trained_model
for file in SOTA CNN_MLP CNN_GCN CNN_RNN_MLP CNN_RNN_GCN Sei_MLP Sei_GCN training_dat fewshot_pathogenic_model sanity_check_data
do
	tar -xvzf ${file}.tar.gz
	mv ${file} trained_model/
done

cd training_data
for file in deepsea_seq_label deepsea_structure_matching_index if_matrix sei_seq_label sei_structure_matching_index whole_gneome_embedding
do
	tar -xvzf ${file}.tar.gz
done
cd ..

cd sanity_check_data
for file in deepsea_data trained_model_prediction
do
	tar -xvzf ${file}.tar.gz
done
cd ..

cd fewshot_pathogenic_model
for file in CNN_RNN_related_pathogenic CNN_related_pathogenic
do
	tar -xvzf ${file}.tar.gz
done

cd ncVar_data
for file in GTEx_eQTL_data hg19_ref_seq ncVarDB_data
do
	tar -xvzf ${file}.tar.gz
done
cd ..

rm *tar.gz 
rm */*tar.gz
