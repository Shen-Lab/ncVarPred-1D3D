import argparse
import numpy as np
from itertools import chain
from Bio import pairwise2

chr_length = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566])
list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])
resolution = 1e5
input_path = 'deepsea_data/'
chr_size = np.asarray(chr_length // resolution + 1, dtype = int)
chr_start = np.zeros(len(chr_length))
for i in np.arange(1, len(chr_length)):
	chr_start[i] = np.sum(chr_size[0:i])

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'seq structure label similarity analysis', description = '')
	parser.add_argument('--hic_name', type = str)
	parser.add_argument('--cellline_name', type = str)
	parser.add_argument('--random_seed', type = int)
	parser.add_argument('--pairwise_resolution', type = int)
	args = parser.parse_args()
	return args

def label_similarity_func(label1, label2):
	return(np.mean(label1 == label2))

def decode_seq_embedding(seq):
	output = ''
	for i in range(seq.shape[1]):
		if(np.sum(seq[:, i]) == 0):
			output = output + '4'
		else:
			output = output + str(np.where(seq[:, i])[0][0])
	return output
	
def seq_similarity_func(seq1, seq2):
	return 2 * np.mean(seq1 == seq2) - 1
	#return pairwise2.align.globalxx(seq1, seq2)[0][2]
	

def main():
	args = parse_arguments()
	hic_name = args.hic_name + '_'
	cellline_name = args.cellline_name
	random_seed = args.random_seed
	pairwise_resolution = args.pairwise_resolution
	output_path = 'data_correction_summary_' + str(int(pairwise_resolution)) + '/pairwise_similarity_whole_' + hic_name + str(int(pairwise_resolution)) + '_' + str(int(random_seed)) + '/'
	if_matrix = np.load('../graph_data/if_matrix_100000/' + hic_name + 'novc_whole_normalized.npy')
	cellline_boolean_index = np.load(input_path + cellline_name + '_feature_index.npy')
	seq = []
	start_index = []
	label = []
	label_cellline = []
	for chr_i in range(len(list_chr)):
		chr_index = list_chr[chr_i]
		seq_temp = np.load(input_path + chr_index + '_seq.npy')
		start_index_temp = np.asarray(np.load(input_path + chr_index + '_start_index.npy') // resolution, dtype = int)
		label_temp = np.array(np.load(input_path + chr_index + '_label.npy'), dtype = int)
		index_temp = np.arange(len(start_index_temp))
		np.random.seed(random_seed)
		np.random.shuffle(index_temp)
		selected_sample_index = np.asarray(index_temp[np.arange(len(start_index_temp)//pairwise_resolution) * pairwise_resolution], dtype = int)
		seq.append(seq_temp[selected_sample_index])
		#for index_i in selected_sample_index:
		#	seq.append(decode_seq_embedding(seq_temp[index_i]))
		start_index.append(start_index_temp[selected_sample_index] + chr_start[chr_i])
		label.append(label_temp[selected_sample_index, :])
		label_cellline.append(label_temp[selected_sample_index][:, cellline_boolean_index])
	seq = list(chain.from_iterable(seq))
	start_index = np.asarray(list(chain.from_iterable(start_index)), dtype = int)
	label = list(chain.from_iterable(label))
	label_cellline = list(chain.from_iterable(label_cellline))
	seq_similarity_output = []
	label_similarity_output = []
	label_cellline_similarity_output = []
	if_similarity_output = []
	for sample_i in range(len(start_index) - 1):
		for sample_j in np.arange(sample_i + 1, len(start_index)):
			if((if_matrix[start_index[sample_i], start_index[sample_j]]) > 0):
				seq_similarity_output.append(seq_similarity_func(seq[sample_i], seq[sample_j]))
				label_similarity_output.append(label_similarity_func(label[sample_i], label[sample_j]))
				label_cellline_similarity_output.append(label_similarity_func(label_cellline[sample_i], label_cellline[sample_j]))
				if_similarity_output.append(if_matrix[start_index[sample_i], start_index[sample_j]])
	seq_similarity_output = np.asarray(seq_similarity_output)
	label_similarity_output = np.asarray(label_similarity_output)
	label_cellline_similarity_output = np.asarray(label_cellline_similarity_output)
	if_similarity_output = np.asarray(if_similarity_output)
	np.save(output_path + hic_name + 'seq_similarity_output.npy', seq_similarity_output)
	np.save(output_path + hic_name + 'label_similarity_output.npy', label_similarity_output)
	np.save(output_path + hic_name + 'label_cellline_similarity_output.npy', label_cellline_similarity_output)
	np.save(output_path + hic_name + 'if_similarity_output.npy', if_similarity_output)

if __name__=='__main__':
	main()

