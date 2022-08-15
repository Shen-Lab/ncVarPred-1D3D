import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model_general import *
from metrics import *
import argparse
from model_resolution100000 import DeepSEA_concatenation

activation = {}
def get_activation(conv1):
	def hook(model, input, output):
		activation[conv1] = output.detach()
	return hook

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'deepsea tunning')
	parser.add_argument('--resolution', type = str)
	parser.add_argument('--model_path', type = str)
	parser.add_argument('--seq_label_path', type = str)
	parser.add_argument('--structure_input_path', type = str)
	parser.add_argument('--structure_input_matching_path', type = str)
	parser.add_argument('--output_path', type = str)
	parser.add_argument('--output_model_name', type = str)
	parser.add_argument('--architecture_name', type = str)
	args = parser.parse_args()
	return args

def main():
	print('testinging start')
	args = parse_arguments()
	resolution = args.resolution
	model_path = args.model_path
	seq_label_path = args.seq_label_path
	structure_input_path = args.structure_input_path
	structure_input_matching_path = args.structure_input_matching_path
	output_path = args.output_path
	output_model_name = args.output_model_name
	architecture_name = args.architecture_name
	motif_result = np.zeros((320, 4, 8))
	if(not (architecture_name in ['deepsea_concatenation', 'deepsea'])):
		print('architecture name wrong')
	bce_loss_threshold = 1e-12
	model = torch.load(model_path, map_location = 'cpu')
	model.conv1.register_forward_hook(get_activation('conv1'))
	model.eval()
	model.cuda()
	test_section_chr = np.load(structure_input_matching_path + 'testing_section_chr.npy')
	test_section_index = np.load(structure_input_matching_path + 'testing_section_index.npy')
	structure_all = np.load(structure_input_path)
	print('testing data loading finished')
	section_size = len(test_section_chr)
	section_index = np.arange(section_size)
	test_pred = np.zeros((455024, 919))
	test_size = test_pred.shape[0]//2
	test_label = np.zeros((test_size, 919))	
	test_sample_index = 0
	seq_count = 0
	activation_value = np.zeros(320)
	for section_i in range(section_size):
		print(section_i)
		test_seq_x = torch.from_numpy(np.load(seq_label_path + test_section_chr[section_index[section_i]] + '_seq_section' + str(int(test_section_index[section_index[section_i]])) + '.npy')).float()
		structure_input_matching_index_npy = np.load(structure_input_matching_path + test_section_chr[section_index[section_i]] + '_section' + str(int(test_section_index[section_index[section_i]])) + '_matching_index.npy')
		test_structure_x = torch.from_numpy(structure_all[np.asarray(structure_input_matching_index_npy, dtype = int), :]).float()
		test_y = torch.from_numpy(np.load(seq_label_path + test_section_chr[section_index[section_i]] + '_label_section' + str(int(test_section_index[section_index[section_i]])) + '.npy')).float()
		test_dataloader = DataLoader(TensorDataset(test_seq_x, test_structure_x, test_y), batch_size = 64, shuffle = False, drop_last=False)		
		for test_batch_i, (test_batch_seq_x, test_batch_structure_x, test_batch_y) in enumerate(test_dataloader):
			test_batch_seq_x_numpy = test_batch_seq_x.numpy()
			test_batch_seq_x_numpy_flipped = np.flip(np.flip(test_batch_seq_x_numpy, 1), 2)
			batch_size = test_batch_seq_x.shape[0]
			test_batch_seq_x = test_batch_seq_x.cuda()
			test_batch_structure_x = test_batch_structure_x.cuda()
			if(architecture_name == 'deepsea_concatenation'):
				test_pred_firsthalf = model(test_batch_seq_x, test_batch_structure_x).cpu().detach().numpy()
			elif(architecture_name == 'deepsea'):
				test_pred_firsthalf = model(test_batch_seq_x).cpu().detach().numpy()
			activation_value_batch = activation['conv1'].cpu().detach().numpy()
			for sample_i in range(batch_size):
				for kernel_i in range(activation_value_batch.shape[1]):
					if(np.max(activation_value_batch[sample_i, kernel_i]) > 0):
						seq_position_index = np.argmax(activation_value_batch[sample_i, kernel_i])
						motif_result[kernel_i] = motif_result[kernel_i] + test_batch_seq_x_numpy[sample_i][:, seq_position_index:int(seq_position_index + 8)]
						seq_count = seq_count + 1
						activation_value[kernel_i] = activation_value[kernel_i] + np.max(activation_value_batch[sample_i, kernel_i])
			test_batch_seq_x = torch.from_numpy(np.flip(np.flip(test_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
			test_batch_seq_x = test_batch_seq_x.cuda()
			if(architecture_name == 'deepsea_concatenation'):
				test_pred_secondhalf = model(test_batch_seq_x, test_batch_structure_x).cpu().detach().numpy()
			elif(architecture_name == 'deepsea'):
				test_pred_secondhalf = model(test_batch_seq_x).cpu().detach().numpy()
			activation_value_batch = activation['conv1'].cpu().detach().numpy()
			for sample_i in range(batch_size):
				for kernel_i in range(activation_value_batch.shape[1]):
					if(np.max(activation_value_batch[sample_i, kernel_i]) > 0):
						seq_position_index = np.argmax(activation_value_batch[sample_i, kernel_i])
						motif_result[kernel_i] = motif_result[kernel_i] + test_batch_seq_x_numpy_flipped[sample_i][:, seq_position_index:int(seq_position_index + 8)]
						seq_count = seq_count + 1
						activation_value[kernel_i] = activation_value[kernel_i] + np.max(activation_value_batch[sample_i, kernel_i])
			test_sample_index = test_sample_index + batch_size
	if(test_sample_index != test_size):
		print('test size error')
	np.save(output_path + output_model_name + '_motif.npy', motif_result / seq_count)
	np.save(output_path + output_model_name + '_motif_activation_value.npy', activation_value / seq_count)

if __name__=='__main__':
	main()


