import numpy as np

input_path = 'GTEx_Analysis_v7_eQTL/'
output_path = 'GTEx_processed/'
#input = open(input_path + 'Cells_Transformed_fibroblasts.v7.egenes.txt').readlines()
#cellline_name = 'IMR90'
input = open(input_path + 'Cells_EBV-transformed_lymphocytes.v7.egenes.txt').readlines()
cellline_name = 'GM12878'
strand_list = []
chr_list = []
coor_list = []
wt_list = []
mt_list = []
qval_list = []
log2_afc_list = []
for i in range(1, len(input)):
	strand_list.append(input[i].split('\t')[5])
	chr_list.append(input[i].split('\t')[13])
	coor_list.append(input[i].split('\t')[14])
	wt_list.append(input[i].split('\t')[15])
	mt_list.append(input[i].split('\t')[16])
	qval_list.append(input[i].split('\t')[28])
	log2_afc_list.append(input[i].split('\t')[30])

egene_boolean = np.array(qval_list, dtype = float) < 0.05
cellline_name = 'GTEx_' + cellline_name
np.save(output_path + cellline_name + '_eqtl_strand.npy', np.array(strand_list)[egene_boolean])
np.save(output_path + cellline_name + '_eqtl_chr.npy', np.array(chr_list)[egene_boolean])
np.save(output_path + cellline_name + '_eqtl_coor.npy', np.array(coor_list)[egene_boolean])
np.save(output_path + cellline_name + '_eqtl_wt.npy', np.array(wt_list)[egene_boolean])
np.save(output_path + cellline_name + '_eqtl_mt.npy', np.array(mt_list)[egene_boolean])
np.save(output_path + cellline_name + '_eqtl_qval.npy', np.array(qval_list)[egene_boolean])
np.save(output_path + cellline_name + '_eqtl_log2_afc.npy', np.array(log2_afc_list)[egene_boolean])



