import numpy as np

input = open('deepsea_supplementary_tables/41592_2015_BFnmeth3547_MOESM649_ESM.csv').readlines()[3:]

chr_list = []
coor_list = []
wt_list = []
mt_list = []
fold_list = []
label_list = []
for row_i in input:
	row_i = row_i.split(',')
	chr_list.append(row_i[1].split('chr')[1])
	coor_list.append(row_i[2])
	wt_list.append(row_i[3])
	mt_list.append(row_i[4])
	fold_list.append(row_i[6])
	label_list.append(row_i[7].split('\n')[0].replace(' ', '_'))

chr_list = np.array(chr_list)
coor_list = np.array(coor_list)
wt_list = np.array(wt_list)
mt_list = np.array(mt_list)
fold_list = np.array(fold_list)
label_list = np.array(label_list)

output_path = 'GRASP_processed/GRASP_'

for label_i in np.unique(label_list):
	for fold_i in np.unique(fold_list):
		selected_boolean = (label_list == label_i) & (fold_list == fold_i)
		section_name = output_path + label_i + '_' + fold_i + '_'
		np.save(section_name + 'chr.npy', chr_list[selected_boolean])
		np.save(section_name + 'coor.npy', coor_list[selected_boolean])
		np.save(section_name + 'wt.npy', wt_list[selected_boolean])
		np.save(section_name + 'mt.npy', mt_list[selected_boolean])

