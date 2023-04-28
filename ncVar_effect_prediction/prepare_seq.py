import numpy as np
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'deepsea tunning')
	parser.add_argument('--seq_chr', type = str)
	parser.add_argument('--seq_coor', type = str)
	parser.add_argument('--seq_wt', type = str)
	parser.add_argument('--seq_mt', type = str)
	#parser.add_argument('--seq_strand', type = str)
	parser.add_argument('--ref_seq_input_path', type = str)
	parser.add_argument('--structure_matching_index_path')
	parser.add_argument('--resolution', type = str)
	parser.add_argument('--output_path', type = str)
	parser.add_argument('--output_name', type = str)
	args = parser.parse_args()
	return args

chr_length = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566, 155270560, 59373566, 16571])
chr_index = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM'])

A = np.array([[1, 0, 0, 0]], dtype = bool)
G = np.array([[0, 1, 0, 0]], dtype = bool)
C = np.array([[0, 0, 1, 0]], dtype = bool)
T = np.array([[0, 0, 0, 1]], dtype = bool)
N = np.array([[0, 0, 0, 0]], dtype = bool)
embedding_dict = {'A': A, 'G': G, 'C': C, 'T': T, 'N': N}

def main():
	args = parse_arguments()
	chr_list = np.load(args.seq_chr)
	coor_list = np.array(np.load(args.seq_coor), dtype = int)
	wt_list = np.load(args.seq_wt)
	mt_list = np.load(args.seq_mt)
	#strand_list = np.load(args.seq_strand)
	ref_seq_input_path = args.ref_seq_input_path
	structure_matching_temp = open(args.structure_matching_index_path).readlines()
	resolution = int(args.resolution)
	output_path = args.output_path
	output_name = args.output_name
	structure_matching_chr = []
	structure_matching_coor = []
	for row in structure_matching_temp:
		structure_matching_chr.append(row.split('\t')[0])
		structure_matching_coor.append(row.split('\t')[1])
	structure_matching_chr = np.array(structure_matching_chr)
	structure_matching_coor = np.array(structure_matching_coor, dtype = int)
	wt_seq_output = np.zeros((len(chr_list), 1000, 4))
	mt_seq_output = np.zeros(wt_seq_output.shape)
	position_index = np.zeros(len(chr_list))
	#if chromosome list looks like chr1, chr1, ..., chr2, ..., remove the chr
	if(len(chr_list[0]) > 2):
		for i in range(len(chr_list)):
			chr_list[i] = chr_list[0][3:]
	chr_temp = chr_list[0]	
	chr_seq = np.load(ref_seq_input_path + 'chr' + chr_temp + '.npy')
	for eqtl_i in range(len(chr_list)):
		if((eqtl_i // 100) == (eqtl_i / 100)):
			print(eqtl_i)
		wt_seq_temp = np.zeros((1000, 4))
		mt_seq_temp = np.zeros(wt_seq_temp.shape)
		if(chr_list[eqtl_i] != chr_temp):
			chr_temp = chr_list[eqtl_i]
			chr_seq = np.load(ref_seq_input_path + 'chr' + chr_temp + '.npy')
		coor_list[eqtl_i] = coor_list[eqtl_i] - 1
		for bp_i in range(len(wt_list[eqtl_i])):	
			if(not np.array_equal(chr_seq[coor_list[eqtl_i] + bp_i, :], embedding_dict[wt_list[eqtl_i][bp_i]][0])):
				print('bp not matched')
				print(chr_seq[coor_list[eqtl_i], :])
				print(embedding_dict[wt_list[eqtl_i][0]][0])
				print(chr_seq[int(coor_list[eqtl_i]-3):int(coor_list[eqtl_i] + 3), :].T)
				print(wt_list[eqtl_i])
				print(mt_list[eqtl_i])
				#print(strand_list[eqtl_i])
		if(chr_seq.shape[0] != chr_length[np.where(chr_index == ('chr' + chr_list[eqtl_i]))[0][0]]):
			print('chr length not matched')
		wt_length = len(wt_list[eqtl_i])
		if(wt_length > 1000):
			print('wt too long')
		wt_start_index = 500 - (wt_length // 2)
		wt_end_index = wt_start_index + wt_length
		wt_left_neigh_length = np.min((wt_start_index, coor_list[eqtl_i]))
		wt_right_neigh_length = np.min((chr_seq.shape[0] - coor_list[eqtl_i], 1000 - wt_end_index))
		wt_seq_temp[int(wt_start_index - wt_left_neigh_length):int(wt_end_index + wt_right_neigh_length), :] = chr_seq[int(coor_list[eqtl_i] - wt_left_neigh_length):int(coor_list[eqtl_i] - wt_left_neigh_length + 1000), :]
		mt_length = len(mt_list[eqtl_i])
		mt_start_index = 500 - (mt_length // 2)
		mt_end_index = mt_start_index + mt_length
		mt_left_neigh_length = np.min((mt_start_index, coor_list[eqtl_i]))
		mt_right_neigh_length = np.min((chr_seq.shape[0] - coor_list[eqtl_i] - wt_length, 1000 - mt_end_index))
		mt_seq_temp[int(mt_start_index - mt_left_neigh_length):int(mt_start_index), :] = chr_seq[int(coor_list[eqtl_i] - mt_left_neigh_length):int(coor_list[eqtl_i]), :]
		mt_seq_embedded = np.zeros((mt_length, 4))
		for bp_i in range(mt_length):
			mt_seq_embedded[bp_i, :] = embedding_dict[mt_list[eqtl_i][bp_i]]
		mt_seq_temp[int(mt_start_index):int(mt_end_index), :] = mt_seq_embedded
		mt_seq_temp[int(mt_end_index):int(mt_end_index + mt_right_neigh_length), :] = chr_seq[int(coor_list[eqtl_i] + wt_length):int(coor_list[eqtl_i] + wt_length + mt_right_neigh_length), :]
		wt_seq_output[eqtl_i, :, :] = wt_seq_temp
		mt_seq_output[eqtl_i, :, :] = mt_seq_temp
		if(((wt_length * mt_length) == 1) & (np.mean(wt_seq_temp[500] == mt_seq_temp[500]) == 1)):
			print('sequence alteration wrong')
			print(eqtl_i)
			print(wt_list[eqtl_i])
			print(mt_list[eqtl_i])
			print(mt_seq_embedded)
			print(mt_start_index)
			print(mt_end_index)
			print(wt_seq_output[eqtl_i, mt_start_index:mt_end_index, :])
			print(mt_seq_output[eqtl_i, mt_start_index:mt_end_index, :])
		position_index[eqtl_i] = np.where((structure_matching_chr == ('chr' + chr_list[eqtl_i])) & (structure_matching_coor == int(float(coor_list[eqtl_i])//1e5*1e5)))[0][0]
	np.save(output_path + output_name + '_wt_seq.npy', wt_seq_output)
	np.save(output_path + output_name + '_mt_seq.npy', mt_seq_output)
	np.save(output_path + output_name + '_structure_matching.npy', np.array(position_index, dtype = int))

if __name__=='__main__':
	main()


