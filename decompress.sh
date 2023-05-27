for file in ncVar_data.tar.gz  sanity_check_data.tar.gz  trained_model.tar.gz  trained_model_Sei_related.tar.gz  training_data.tar.gz
do
	tar -xvzf ${file}
done

cd ncVar_data
for file in GTEx_eQTL_data.tar.gz hg19_ref_seq.tar.gz ncVarDB_data.tar.gz
do
	tar -xvzf ${file}
done
cd ..

cd sanity_check_data
for file in deepsea_data.tar.gz trained_model_prediction.tar.gz
do
	tar -xvzf ${file}
done
cd ..

mkdir -p trained_model
cd trained_model
for file in CNN_GCN.tar.gz CNN_MLP.tar.gz CNN_RNN_GCN.tar.gz CNN_RNN_MLP.tar.gz SOTA.tar.gz fewshot_pathogenic_model.tar.gz
do
	mv ../${file} .
	tar -xvzf ${file}
done
cd ..

cd trained_model_Sei_related
for file in Sei_GCN.tar.gz  Sei_MLP.tar.gz
do
	tar -xvzf ${file}
done
cd ..
cd training_data
for file in deepsea_seq_label.tar.gz if_matrix.tar.gz sei_seq_label.tar.gz whole_genome_embedding.tar.gz deepsea_structure_matching_index.tar.gz sei_structure_matching_index.tar.gz
do
	tar -xvzf ${file}
done
cd ..

mv trained_model_Sei_related/Sei_GCN trained_model/
mv trained_model_Sei_related/Sei_MLP trained_model/

rm *.tar.gz
rm */*.tar.gz
