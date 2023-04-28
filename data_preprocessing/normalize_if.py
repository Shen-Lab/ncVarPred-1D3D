import scipy.sparse as sp
import os
import numpy as np

def normalize(adj):
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

hic_list = eval(open('hic_list_whole.txt', 'r').read())

path = 'if_matrix_novc_whole/'

for resolution_i in ['100000', '500000', '1000000']:
	for key_index_i in range(len(list(hic_list.keys()))):
		hic_list_temp = hic_list[list(hic_list.keys())[key_index_i]]
		for hic_i in range(len(hic_list_temp)):
			hic_name_i = hic_list_temp[hic_i]
			print(hic_name_i)
			input = np.load(path + 'if_matrix_' + resolution_i + '_result/' + hic_name_i + '_novc_whole.npy')
			adj = normalize(input + sp.eye(input.shape[0]))
			np.save(path + 'if_matrix_' + resolution_i + '/' + hic_name_i + '_novc_whole_normalized.npy', adj)
			
