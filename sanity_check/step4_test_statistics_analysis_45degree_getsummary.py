import numpy as np
from scipy.stats import spearmanr

hic_list = ['ENCFF014VMM', 'ENCFF928NJV', 'ENCFF013TGD']
random_seed_list = np.asarray(np.arange(100) * 100, dtype = str)
pairwise_resolution = '100'
path = 'data_correction_summary_' + pairwise_resolution + '/45degree_statistics_summary/'

cutoff = '0.2'
selected_boolean_name = ['_lshl', '_central', '_hsll']

for hic_i in hic_list:
	statistics = np.zeros((100, 2, 3))
	pvalue = np.zeros(statistics.shape)
	for seed_i in range(len(random_seed_list)):
		input = np.load(path + hic_i + '_' + pairwise_resolution + '_' + random_seed_list[seed_i] + '_similarity_summary.npy', allow_pickle = True).item()
		for selection_i in range(len(selected_boolean_name)):
			key_name = cutoff + selected_boolean_name[selection_i]
			statistics[seed_i, 0, selection_i], pvalue[seed_i, 0, selection_i] = spearmanr(input[key_name]['label'], input[key_name]['sota'])
			statistics[seed_i, 1, selection_i], pvalue[seed_i, 1, selection_i] = spearmanr(input[key_name]['label'], input[key_name]['cellline'])
	np.save(path + hic_i + '_spearman_statistics.npy', statistics)
	np.save(path + hic_i + '_spearman_pvalue.npy', pvalue)


