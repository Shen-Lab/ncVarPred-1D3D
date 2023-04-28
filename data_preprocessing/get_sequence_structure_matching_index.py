import numpy as np

def parse_arguments():
	parser.add_argument('--seq_chr', type = str)
	parser.add_argument('--seq_coor', type = str)
	parser.add_argument('--resolution', type = int)
	parser.add_argument('--structure_matching_index_path')
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	chr_list = np.load(args.seq_chr)
	coor_list = np.array(np.load(args.seq_coor), dtype = int)
	resolution = args.resolution
	structure_matching_temp = open(args.structure_matching_index_path).readlines()
	structure_matching_chr = []
	structure_matching_coor = []
	for row in structure_matching_temp:
		structure_matching_chr.append(row.split('\t')[0])
		structure_matching_coor.append(row.split('\t')[1])
	structure_matching_chr = np.array(structure_matching_chr)
	structure_matching_coor = np.array(structure_matching_coor, dtype = int)
	position_index = np.zeros(len(chr_list))
	for seq_i in range(len(chr_list)):
		position_index[eqtl_i] = np.where((structure_matching_chr == ('chr' + chr_list[eqtl_i])) & (structure_matching_coor == int(float(coor_list[eqtl_i]) // resolution * resolution)))[0][0]
	np.save(output_path + output_name + '_structure_matching.npy', np.array(position_index, dtype = int))

if __name__=='__main__':
	main()
