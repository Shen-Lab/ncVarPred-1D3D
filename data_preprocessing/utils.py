import numpy as np
import wget
import os

"""
This will download the chromatin structure data, .hic, process the interaction frequency.
This code require the juicer tools.
chr_index_list = np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M'])))
"""

def download_hic(path, hic_name):
	"""
	This function download the chromatin structure data, .hic file.
	'path' is the folder to save the downloaded experiment data.
	'hic_name' is the ENCODE project name for the wanted hic experiment.
	"""
	wget.download('https://www.encodeproject.org/files/' + hic_name + '/@@download/' + hic_name + '.hic', out = path)

def extract_if_from_hic(input_path, juicer_path, output_path, hic_name, resolution, normalize_method, chr_index_list):
	"""
	By calling juicer tools, extract the interaction frequency from the .hic file. multiple interaction frequency resolution, normalizion methods can be selected.
	You may get more information about juicer usage by checking the github https://github.com/aidenlab/juicer/wiki/Juicer-Tools-Quick-Start
	'input_path' is the working directory, used to save and process the .hic file.
	'juicer_path' is the path to the juicer tools.
	'output_path' is the path you save the processed the interaction frequency matrix
	'hic_name' is the ENCODE project experiment name for the analyzed .hic data.
	'resolution' is the chromatin structure resolution. It can be determined by the .hic file. Usually choose 100000 or 1000000.
	'normalized_method' is the normalization method we want to use, VC, NONE, etc.
	"""
	output_path = output_path + 'if_matrix_' + resolution + '/'
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	for chr_i in range(len(chr_index_list)):
		chr_i_name = chr_index_list[chr_i]
		for chr_j in range(chr_i, len(chr_index_list)):
			chr_j_name = chr_index_list[chr_j]
			os.system('java -jar ' + juicer_path + ' dump observed ' + normalize_method + ' ' + input_path + hic_name + '.hic ' + chr_i_name + ' ' + chr_j_name + ' BP ' + resolution + ' ' + output_path + hic_name + '_' + chr_i_name + '_' + chr_j_name + '.txt')


def if_txt_to_npy(input_path, hic_name, resolution, chr_length, output_path, chr_index_list):
	"""
	This function convert the juicer tool output text file into numpy array. The output is a chromatin-chromatin pairwise result.
	'input_path' is the folder where the processed chromatin structure are saved.
	'hic_name' is the encode project experiment name for the given .hic.
	'resolution' is the selected chromatin structure resolution.
	'chr_length' is the length of each chromosome.
	'output_path' is the folder where the processed numpy.array are saved.
	"""
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	chr_size = np.array(chr_length // int(resolution) + 1, dtype = int)
	for chr_i in range(len(chr_index_list)):
		for chr_j in range(chr_i, len(chr_index_list)):
			output = np.zeros((chr_size[chr_i], chr_size[chr_j]))
			if(os.path.isfile(input_path + hic_name + '_' + chr_index_list[chr_i] + '_' + chr_index_list[chr_j] + '.txt')):
				input = open(input_path + hic_name + '_' + chr_index_list[chr_i] + '_' + chr_index_list[chr_j] + '.txt').readlines()
				coordinate_1 = []
				coordinate_2 = []
				counts = []
				for i in range(len(input)):
					if_float = float(input[i].split('\t')[2].split('\n')[0])
					if(not np.isnan(if_float)):
						coordinate_1.append(int(input[i].split('\t')[0]) // int(resolution))
						coordinate_2.append(int(input[i].split('\t')[1]) // int(resolution))
						counts.append(if_float)
				coordinate_1 = np.asarray(coordinate_1)
				coordinate_2 = np.asarray(coordinate_2)
				counts = np.asarray(counts)
				if(len(counts) == 0):
					print(chr_index_list[chr_i] + '_' + chr_index_list[chr_j])
				elif(np.min(counts) == 0):
					print(chr_index_list[chr_i] + '_' + chr_index_list[chr_j])
				for i in range(len(counts)):
					output[int(coordinate_1[i]), int(coordinate_2[i])] = counts[i]
					if(chr_i == chr_j):
						output[int(coordinate_2[i]), int(coordinate_1[i])] = counts[i]
			else:
				print(hic_name + chr_index_list[chr_i] + '_' + chr_index_list[chr_j])
			np.save(output_path + hic_name + '_' + chr_index_list[chr_i] + '_' + chr_index_list[chr_j] + '.npy', output)

def merge_if_npy(input_path, hic_name, resolution, chr_length, output_path, chr_index_list):
	"""
	This function merge the chromatin-chromatin pairwise result into a whole genome numpy array.
	'input_path' is the processed chromatin interaction frequency numpy array are saved.
	'hic_name' is the encode project experiment name for the given .hic.
	'resolution' is the selected chromatin structure resolution.
	'output_path' is where the merged whole interaction frequency is saved.
	"""
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	chr_size = np.asarray(chr_length // int(resolution) + 1, dtype = int)
	chr_start = np.zeros(len(chr_length))
	for i in np.arange(1, len(chr_length)):
		chr_start[i] = np.sum(chr_size[0:i])
	total_length = int(np.sum(chr_size))
	output = np.zeros((total_length, total_length))
	for chr_i in range(len(chr_index_list)):
		for chr_j in  range(len(chr_index_list)):
			if(chr_i <= chr_j):
				input = np.load(input_path + hic_name + '_' + chr_index_list[chr_i] + '_' + chr_index_list[chr_j] + '.npy')
			else:
				input = np.load(input_path + hic_name + '_' + chr_index_list[chr_j] + '_' + chr_index_list[chr_i] + '.npy').T
			output[int(chr_start[chr_i]):int(chr_start[chr_i]+chr_size[chr_i]), int(chr_start[chr_j]):int(chr_start[chr_j]+chr_size[chr_j])] = input
	np.save(output_path + hic_name + '_novc_whole.npy', output)

