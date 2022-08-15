import numpy as np
from scipy import stats
import argparse

violin_label_input_name = ['', 'cellline_specific']
diff_cutoff_list = np.arange(1, 16) / 20

def prepare_percentile(if_input, seq_input, label_input):
	percentile_cutoff_size = 100
	label_percentile_cutoff = np.zeros(percentile_cutoff_size)
	seq_percentile_cutoff = np.zeros(percentile_cutoff_size)
	label_sorted = np.sort(label_input)
	seq_sorted = np.sort(seq_input)
	pair_size = len(label_input)
	for cutoff_i in range(percentile_cutoff_size):
		seq_percentile_cutoff[cutoff_i] = seq_sorted[int(cutoff_i * pair_size / percentile_cutoff_size)]
		label_percentile_cutoff[cutoff_i] = label_sorted[int(cutoff_i * pair_size / percentile_cutoff_size)]
	seq_percentile = np.zeros(pair_size)
	label_percentile = np.zeros(pair_size)
	for pair_i in range(pair_size):
		seq_percentile[pair_i] = np.mean(seq_input[pair_i] >= seq_percentile_cutoff)
		label_percentile[pair_i] = np.mean(label_input[pair_i] >= label_percentile_cutoff)
	return seq_percentile, label_percentile

def split_quadrant(if_input, seq_percentile, label_percentile, diff_cutoff):
	low_seq_high_label_boolean = ((label_percentile - seq_percentile) >= diff_cutoff)
	high_seq_low_label_boolean = ((seq_percentile - label_percentile) >= diff_cutoff)
	general_boolean = np.invert(low_seq_high_label_boolean | high_seq_low_label_boolean)
	return [if_input[low_seq_high_label_boolean], if_input[general_boolean], if_input[high_seq_low_label_boolean]]

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'seq structure label similarity analysis', description = '')
	parser.add_argument('--hic_name', type = str)
	parser.add_argument('--pairwise_resolution', type = str)
	parser.add_argument('--random_seed', type = str)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	hic_name = args.hic_name
	pairwise_resolution = args.pairwise_resolution
	random_seed = args.random_seed
	path = 'data_correction_summary_' + pairwise_resolution + '/' + 'pairwise_similarity_whole_' + hic_name + '_' + pairwise_resolution + '_' + random_seed + '/' + hic_name
	output_path = 'data_correction_summary_' + pairwise_resolution + '/45degree_statistics_summary/' + hic_name + '_' + pairwise_resolution + '_' + random_seed + '_'
	if_similarity = np.load(path + '_if_similarity_output.npy')
	label_cellline_similarity = np.load(path + '_label_cellline_similarity_output.npy')
	label_similarity = np.load(path + '_label_similarity_output.npy')
	violin_plot_seq_input = np.load(path + '_seq_similarity_output.npy')
	violin_plot_label_input = [label_similarity, label_cellline_similarity]
	t_test_vs_rest_pvalue = np.zeros((len(violin_label_input_name), len(diff_cutoff_list), 2)) - 1
	count = np.zeros((len(violin_label_input_name), len(diff_cutoff_list), 3)) - 1
	t_test_vs_central_pvalue = np.zeros(t_test_vs_rest_pvalue.shape) - 1
	t_test_vs_central_statistics = np.zeros(t_test_vs_rest_pvalue.shape) - 1
	ks_test_vs_central_pvalue = np.zeros(t_test_vs_rest_pvalue.shape) - 1
	ks_test_vs_central_statistics = np.zeros(t_test_vs_rest_pvalue.shape) - 1
	for violin_label_input_list_index in range(len(violin_label_input_name)):
		print(violin_label_input_name[violin_label_input_list_index])
		seq_percentile, label_percentile = prepare_percentile(if_similarity, violin_plot_seq_input, violin_plot_label_input[violin_label_input_list_index])
		for diff_cutoff_i in range(len(diff_cutoff_list)):
			print(diff_cutoff_i)
			violinplot_input = split_quadrant(if_similarity, seq_percentile, label_percentile, diff_cutoff_list[diff_cutoff_i])
			if(np.min((len(violinplot_input[0]), len(violinplot_input[1]))) > 0):
				for i in range(3):
					count[violin_label_input_list_index, diff_cutoff_i, i] = len(violinplot_input[i])
				ks_test_vs_central_statistics[violin_label_input_list_index, diff_cutoff_i, 0], ks_test_vs_central_pvalue[violin_label_input_list_index, diff_cutoff_i, 0] = stats.ks_2samp(violinplot_input[0], violinplot_input[1], alternative = 'less')
				t_test_vs_central_statistics[violin_label_input_list_index, diff_cutoff_i, 0], t_test_vs_central_pvalue[violin_label_input_list_index, diff_cutoff_i, 0] = stats.ttest_ind(violinplot_input[0], violinplot_input[1], equal_var=False)
				ks_test_vs_central_statistics[violin_label_input_list_index, diff_cutoff_i, 1], ks_test_vs_central_pvalue[violin_label_input_list_index, diff_cutoff_i, 1] = stats.ks_2samp(violinplot_input[1], violinplot_input[2], alternative = 'less')
				t_test_vs_central_statistics[violin_label_input_list_index, diff_cutoff_i, 1], t_test_vs_central_pvalue[violin_label_input_list_index, diff_cutoff_i, 1] = stats.ttest_ind(violinplot_input[1], violinplot_input[2], equal_var=False)
	np.save(output_path + 'count.npy', count)
	np.save(output_path + 'ks_test_vs_central_statistics.npy', ks_test_vs_central_statistics)
	np.save(output_path + 'ks_test_vs_central_pvalue.npy', ks_test_vs_central_pvalue)
	np.save(output_path + 't_test_vs_central_pvalue.npy', t_test_vs_central_pvalue)
	np.save(output_path + 't_test_vs_central_statistics.npy', t_test_vs_central_statistics)

if __name__=='__main__':
	main()

