import numpy as np

hic_list = {'GM12878': ['ENCFF227XJZ', 'ENCFF014VMM', 'ENCFF563XES', 'ENCFF482LGO', 'ENCFF053BXY', 'ENCFF812THZ', 'ENCFF688KOY', 'ENCFF777KBU', 'ENCFF065LSP', 'ENCFF718AWL', 'ENCFF632MFV', 'ENCFF355OWW', 'ENCFF514XWQ', 'ENCFF223UBX', 'ENCFF799QGA', 'ENCFF473CAA'], 'IMR90': ['ENCFF999YXX', 'ENCFF043EEE', 'ENCFF029MPB', 'ENCFF894GLR', 'ENCFF997RGL', 'ENCFF920CJR', 'ENCFF928NJV', 'ENCFF303PCK', 'ENCFF366ERB'], 'K562': ['ENCFF406HHC', 'ENCFF013TGD', 'ENCFF996XEO', 'ENCFF464KRA', 'ENCFF929RPW', 'ENCFF097SKJ']}

random_seed_list = ['0', '100', '200']

pairwise_resolution = '50'
path = 'data_correction_summary_' + pairwise_resolution + '/45degree_statistics_summary/'

for cellline_i in range(len(list(hic_list.keys()))):
	cellline_name = list(hic_list.keys())[cellline_i]
	cellline_replicate_num = len(hic_list[cellline_name])
	input_temp = np.load(path + hic_list[cellline_name][0] + '_' + pairwise_resolution + '_' + random_seed_list[0] + '_wilcoxon_rank_sum_pvalue.npy')
	output_wilcoxon_rank_sum_statistics = np.zeros((cellline_replicate_num, len(random_seed_list), input_temp.shape[0], input_temp.shape[1]))
	output_wilcoxon_rank_sum_pvalue = np.zeros(output_wilcoxon_rank_sum_statistics.shape)
	input_temp = np.load(path + hic_list[cellline_name][0] + '_' + pairwise_resolution + '_' + random_seed_list[0] + '_count.npy')
	output_count = np.zeros((cellline_replicate_num, len(random_seed_list), input_temp.shape[0], input_temp.shape[1]))
	for random_seed_i in range(len(random_seed_list)):
		for replicate_i in range(cellline_replicate_num):
			replicate_name = hic_list[cellline_name][replicate_i]
			output_count[replicate_i, random_seed_i] = np.load(path + replicate_name + '_' + pairwise_resolution + '_' + random_seed_list[random_seed_i] + '_count.npy')
			output_wilcoxon_rank_sum_statistics[replicate_i, random_seed_i] = np.load(path + replicate_name + '_' + pairwise_resolution + '_' + random_seed_list[random_seed_i] + '_wilcoxon_rank_sum_statistics.npy')
			output_wilcoxon_rank_sum_pvalue[replicate_i, random_seed_i] = np.load(path + replicate_name + '_' + pairwise_resolution + '_' + random_seed_list[random_seed_i] + '_wilcoxon_rank_sum_pvalue.npy')
	np.save(path + cellline_name + '_count.npy', output_count)
	np.save(path + cellline_name + '_wilcoxon_rank_sum_statistics.npy', output_wilcoxon_rank_sum_statistics)
	np.save(path + cellline_name + '_wilcoxon_rank_sum_pvalue.npy', output_wilcoxon_rank_sum_pvalue)
