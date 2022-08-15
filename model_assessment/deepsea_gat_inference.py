import numpy as np
from sklearn import metrics
import torch
from model_gat import DeepSEA_concatenation
from metrics import *
from torch.utils.data import TensorDataset, DataLoader
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'deepsea tunning')
	parser.add_argument('--model_path', type = str)
	parser.add_argument('--seq_label_path', type = str)
	parser.add_argument('--structure_input_path', type = str)
	parser.add_argument('--structure_input_matching_path', type = str)
	parser.add_argument('--output_path', type = str)
	parser.add_argument('--output_model_name', type = str)
	parser.add_argument('--node_feature_type', type = str, default = 'allones')
	parser.add_argument('--architecture_name', type = str)
	args = parser.parse_args()
	return args

def khop_neighbor(adj_input, node_list, k):
	selected_node_list = np.zeros(adj_input.shape[0])
	selected_node_list[node_list] = 1
	for k_i in range(k):
		selected_node_list_k_i = np.where(selected_node_list)[0]
		for node_i in selected_node_list_k_i:
			selected_node_list[adj_input[node_i, :] > 2e-3] = 1
	return np.array(selected_node_list, dtype = bool)

def main():
	print('testinging start')
	args = parse_arguments()
	model_path = args.model_path
	seq_label_path = args.seq_label_path
	structure_input_path = args.structure_input_path
	structure_input_matching_path = args.structure_input_matching_path
	output_path = args.output_path
	output_model_name = args.output_model_name
	node_feature_type = args.node_feature_type
	architecture_name = args.architecture_name
	if(not (architecture_name in ['deepsea_concatenation', 'deepsea'])):
		print('architecture name wrong')
	structure_all = np.load(structure_input_path)
	if(node_feature_type == 'dnabert'):
		node_feature = np.load('dnabert_embedded_mean.npy')
	elif(node_feature_type == 'allones'):
		node_feature = np.zeros((structure_all.shape[0], 768)) + 1
	bce_loss_threshold = 1e-12
	model = torch.load(model_path, map_location = 'cpu')
	model.eval()
	model.cuda()
	khop = 2
	valid_seq_x = torch.from_numpy(np.load(seq_label_path + 'seq_validation.npy')).float()
	valid_structure_matching_index = np.asarray(np.load(structure_input_matching_path + 'validation_matching_index.npy'), dtype = int)
	valid_khop_node_boolean = khop_neighbor(structure_all, valid_structure_matching_index, khop)
	valid_khop_node_index = np.where(valid_khop_node_boolean)[0]
	valid_node_feature = torch.from_numpy(node_feature[valid_khop_node_boolean]).float().cuda()
	valid_y = torch.from_numpy(np.load(seq_label_path + 'label_validation.npy')).float()
	valid_node_selection_index = np.zeros((len(valid_structure_matching_index), len(valid_khop_node_index)))
	for sample_i in range(len(valid_structure_matching_index)):
		valid_node_selection_index[sample_i, np.where(valid_khop_node_index == valid_structure_matching_index[sample_i])[0][0]] = 1
	valid_structure_input = structure_all[valid_khop_node_boolean][:, valid_khop_node_boolean]
	valid_structure_input = torch.from_numpy(valid_structure_input).float().cuda()
	valid_node_selection_index = torch.from_numpy(valid_node_selection_index).float()
	valid_dataloader = DataLoader(TensorDataset(valid_seq_x, valid_node_selection_index, valid_y), batch_size = 50, shuffle = False, drop_last=False)
	valid_pred = np.zeros((8000, 919))
	valid_size = valid_pred.shape[0] // 2
	valid_label = np.zeros((valid_size, 919))
	valid_sample_index = 0
	test_section_chr = np.load(structure_input_matching_path + 'testing_section_chr.npy')
	test_section_index = np.load(structure_input_matching_path + 'testing_section_index.npy')
	print('testing data loading finished')
	section_size = len(test_section_chr)
	section_index = np.arange(section_size)
	test_pred = np.zeros((455024, 919))
	test_size = test_pred.shape[0]//2
	test_label = np.zeros((test_size, 919))	
	for valid_batch_i, (valid_batch_seq_x, valid_batch_node_selection_index, valid_batch_y) in enumerate(valid_dataloader):
		batch_size = valid_batch_seq_x.shape[0]
		valid_batch_seq_x = valid_batch_seq_x.cuda()
		valid_batch_node_selection_index = valid_batch_node_selection_index.cuda()
		if(architecture_name == 'deepsea_concatenation'):
			valid_pred_firsthalf = model(valid_batch_seq_x, valid_node_feature, valid_structure_input, valid_batch_node_selection_index).cpu().detach().numpy()
		elif(architecture_name == 'deepsea'):
			valid_pred_firsthalf = model(valid_batch_seq_x).cpu().detach().numpy()
		valid_pred[int(valid_sample_index):int(valid_sample_index + batch_size), :] = valid_pred_firsthalf
		valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
		valid_batch_seq_x = valid_batch_seq_x.cuda()
		if(architecture_name == 'deepsea_concatenation'):
			valid_pred_secondhalf = model(valid_batch_seq_x, valid_node_feature, valid_structure_input, valid_batch_node_selection_index).cpu().detach().numpy()
		elif(architecture_name == 'deepsea'):
			valid_pred_secondhalf = model(valid_batch_seq_x).cpu().detach().numpy()
		valid_pred[int(valid_sample_index + valid_size):int(valid_sample_index + batch_size + valid_size), :] = valid_pred_secondhalf
		valid_label[int(valid_sample_index):int(valid_sample_index + batch_size)] = valid_batch_y
		valid_sample_index = valid_sample_index + batch_size
	if(valid_sample_index != valid_size):
		print('validation size error')
	valid_loss_bce = bce_loss(valid_pred[0:valid_size], valid_label, bce_loss_threshold) + bce_loss(valid_pred[valid_size:], valid_label, bce_loss_threshold)
	valid_loss_bce = valid_loss_bce / 2
	valid_pred = (valid_pred[0:valid_size] + valid_pred[valid_size:]) / 2
	valid_auroc = []
	valid_auprc = []
	for i in range(919):
		valid_auroc.append(get_auroc(valid_pred[:, i], valid_label[:, i]))
		valid_auprc.append(get_auprc(valid_pred[:, i], valid_label[:, i]))
	test_sample_index = 0
	for section_i in range(section_size):
		print(section_i)
		test_seq_x = torch.from_numpy(np.load(seq_label_path + test_section_chr[section_index[section_i]] + '_seq_section' + str(int(test_section_index[section_index[section_i]])) + '.npy')).float()
		test_structure_matching_index = torch.from_numpy(np.array(np.load(structure_input_matching_path + test_section_chr[section_index[section_i]] + '_section' + str(int(test_section_index[section_index[section_i]])) + '_matching_index.npy'), dtype = int)).float()
		test_y = torch.from_numpy(np.load(seq_label_path + test_section_chr[section_index[section_i]] + '_label_section' + str(int(test_section_index[section_index[section_i]])) + '.npy')).float()
		test_dataloader = DataLoader(TensorDataset(test_seq_x, test_structure_matching_index, test_y), batch_size = 64, shuffle = False, drop_last=False)		
		for test_batch_i, (test_batch_seq_x, test_batch_structure_matching_index, test_batch_y) in enumerate(test_dataloader):
			batch_size = test_batch_seq_x.shape[0]
			test_batch_seq_x = test_batch_seq_x.cuda()
			test_batch_structure_matching_index = test_batch_structure_matching_index.detach().numpy()
			test_batch_structure_matching_index = np.array(test_batch_structure_matching_index, dtype = int)
			test_batch_khop_node_boolean = khop_neighbor(structure_all, test_batch_structure_matching_index, khop)
			test_batch_khop_node_index = np.where(test_batch_khop_node_boolean)[0]
			test_batch_node_selection_index = np.zeros((len(test_batch_structure_matching_index), len(test_batch_khop_node_index)))
			for sample_i in range(len(test_batch_structure_matching_index)):
				test_batch_node_selection_index[sample_i, np.where(test_batch_khop_node_index == test_batch_structure_matching_index[sample_i])[0][0]] = 1
			test_batch_node_selection_index = torch.from_numpy(test_batch_node_selection_index).float().cuda()
			test_batch_structure_input = structure_all[test_batch_khop_node_boolean][:, test_batch_khop_node_boolean]
			test_batch_structure_input = torch.from_numpy(test_batch_structure_input).float().cuda()
			test_batch_node_input = torch.from_numpy(node_feature[test_batch_khop_node_boolean]).float().cuda()
			test_batch_y = test_batch_y.cuda()
			if(architecture_name == 'deepsea_concatenation'):
				test_pred_firsthalf = model(test_batch_seq_x, test_batch_node_input, test_batch_structure_input, test_batch_node_selection_index).cpu().detach().numpy()
			elif(architecture_name == 'deepsea'):
				test_pred_firsthalf = model(test_batch_seq_x).cpu().detach().numpy()
			test_pred[int(test_sample_index):int(test_sample_index + batch_size), :] = test_pred_firsthalf
			test_batch_seq_x = torch.from_numpy(np.flip(np.flip(test_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
			test_batch_seq_x = test_batch_seq_x.cuda()
			if(architecture_name == 'deepsea_concatenation'):
				test_pred_secondhalf = model(test_batch_seq_x, test_batch_node_input, test_batch_structure_input, test_batch_node_selection_index).cpu().detach().numpy()
			elif(architecture_name == 'deepsea'):
				test_pred_secondhalf = model(test_batch_seq_x).cpu().detach().numpy()
			test_pred[int(test_sample_index + test_size):int(test_sample_index + batch_size + test_size), :] = test_pred_secondhalf
			test_label[int(test_sample_index):int(test_sample_index + batch_size)] = test_batch_y.cpu().detach().numpy()
			test_sample_index = test_sample_index + batch_size
	if(test_sample_index != test_size):
		print('test size error')
	test_pred = (test_pred[0:test_size] + test_pred[test_size:]) / 2
	test_auroc = []
	test_auprc = []
	for i in range(919):
		test_auroc.append(get_auroc(test_pred[:, i], test_label[:, i]))
		test_auprc.append(get_auprc(test_pred[:, i], test_label[:, i]))
	#np.save(output_path + output_model_name + '_validpred.npy', valid_pred)
	#np.save(output_path + output_model_name + '_testpred.npy', test_pred)
	#np.save(output_path + output_model_name + '_testlabel.npy', test_label)
	np.save(output_path + output_model_name + '_bceloss' + str(round(valid_loss_bce, 7)) + '_auroc' + str(round(np.nanmean(np.array(valid_auroc)), 5)) + '_auprc' + str(round(np.nanmean(np.array(valid_auprc)), 5)) + '_auroc' + str(round(np.nanmean(np.array(test_auroc)), 5)) + '_auprc' + str(round(np.nanmean(np.array(test_auprc)), 5)) + '.npy', np.zeros(1))
	np.save(output_path + output_model_name + '_auroc.npy', np.array(test_auroc, dtype = float))
	np.save(output_path + output_model_name + '_auprc.npy', np.array(test_auprc, dtype = float))

if __name__=='__main__':
	main()


