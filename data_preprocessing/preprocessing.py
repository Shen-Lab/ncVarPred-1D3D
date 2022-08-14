from utils import *

chr_length_hg19 = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566, 155270560, 59373566, 16571])
chr_length_hg38 = np.array([248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895, 57227415, 16569])
chr_index_list = np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M'])))

hic_list = eval(open('hic_list_whole.txt', 'r').read())

for key_i in list(hic_list.keys()):
	for hic_i in hic_list[key_i]:
		download_hic('hic_downloaded/', hic_i)
		for resolution_i in ['100000', '500000', '1000000']:
		for resolution_i in ['100000']:
			print(hic_i)
			extract_if_from_hic('hic_downloaded/', './juicer_tools_1.22.01.jar', './', hic_i, resolution_i, 'NONE', chr_index_list)
			if_txt_to_npy('if_matrix_' + resolution_i + '/', hic_i, resolution_i, chr_length_hg19, 'if_matrix_' + resolution_i + '/', chr_index_list)
			merge_if_npy('if_matrix_' + resolution_i + '/', hic_i, resolution_i, chr_length_hg19, 'if_matrix_' + resolution_i + '_result/', chr_index_list) 

