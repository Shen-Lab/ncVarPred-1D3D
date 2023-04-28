import numpy as np
from scipy import stats
import argparse

diff_cutoff_list = [0.2]

def prepare_percentile(seq_input, label_input):
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

def split_quadrant(seq_percentile, label_percentile, diff_cutoff):
	low_seq_high_label_boolean = ((label_percentile - seq_percentile) >= diff_cutoff)
	high_seq_low_label_boolean = ((seq_percentile - label_percentile) >= diff_cutoff)
	general_boolean = np.invert(low_seq_high_label_boolean | high_seq_low_label_boolean)
	return low_seq_high_label_boolean, general_boolean, high_seq_low_label_boolean

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
	label_similarity = np.load(path + '_label_similarity_output.npy')
	sota_similarity = np.load(path + '_sota_pred_abs_similarity_output.npy')
	cellline_similarity = np.load(path + '_concate_pred_abs_similarity_output.npy')
	seq_similarity = np.load(path + '_seq_similarity_output.npy')
	output = {}
	seq_percentile, label_percentile = prepare_percentile(seq_similarity, label_similarity)
	selected_boolean_name = ['_lshl', '_central', '_hsll']
	for diff_cutoff_i in range(len(diff_cutoff_list)):
		print(diff_cutoff_i)
		selected_boolean = split_quadrant(seq_percentile, label_percentile, diff_cutoff_list[diff_cutoff_i])
		for selection_i in range(len(selected_boolean_name)):
			key = str(diff_cutoff_list[diff_cutoff_i]) + selected_boolean_name[selection_i]
			output[key] = {}
			output[key]['if'] = if_similarity[selected_boolean[selection_i]]
			output[key]['label'] = label_similarity[selected_boolean[selection_i]]
			output[key]['sota'] = sota_similarity[selected_boolean[selection_i]]
			output[key]['cellline'] = cellline_similarity[selected_boolean[selection_i]]
		
	np.save(output_path + 'similarity_summary.npy', output)
			
if __name__=='__main__':
	main()

