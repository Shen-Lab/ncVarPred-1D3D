import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import GraphConvolution
#from layers import GraphAttentionLayer, SpGraphAttentionLayer
from model_gat import *
from model_general import *
from sklearn import metrics
import argparse
from torch.utils.data import TensorDataset, DataLoader
from libauc.optimizers import SOAP_SGD, SOAP_ADAM
from libauc.losses import APLoss_SH

def get_auprc(preds, obs):
	precision, recall, thresholds  = metrics.precision_recall_curve(obs, preds)
	auprc = metrics.auc(recall, precision)
	return auprc

def khop_neighbor(adj_input, node_list, k):
	selected_node_list = np.zeros(adj_input.shape[0])
	selected_node_list[node_list] = 1
	for k_i in range(k):
		selected_node_list_k_i = np.where(selected_node_list)[0]
		for node_i in selected_node_list_k_i:
			selected_node_list[adj_input[node_i, :] > 1e-4] = 1
	return np.array(selected_node_list, dtype = bool)

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'concatenation architecture tunning')
	parser.add_argument('--structure_name', type = str)
	parser.add_argument('--resolution', type = str)
	parser.add_argument('--learning_rate', type = float)
	parser.add_argument('--warmstart_boolean', type = str)
	parser.add_argument('--retrain_boolean', type = str)
	parser.add_argument('--model_input_path', type = str, default = '')
	parser.add_argument('--model_output_path', type = str)
	parser.add_argument('--train_batch_size', type = int)
	parser.add_argument('--lambda_l1', type = str, default = '0')
	parser.add_argument('--lambda_l2', type = str, default = '0')
	parser.add_argument('--node_feature_type', type = str, default = 'allones')
	parser.add_argument('--loss_type', type = str, default = 'BCE')
	parser.add_argument('--sh_margin', type = str, default = '0.9')
	args = parser.parse_args()
	return args

def main():
	print('training start')
	args = parse_arguments()
	structure_name = args.structure_name
	resolution = args.resolution
	lr = args.learning_rate
	warmstart_boolean = args.warmstart_boolean
	retrain_boolean = args.retrain_boolean
	model_input_path = args.model_input_path
	model_output_path = args.model_output_path
	train_batch_size = args.train_batch_size
	lambda_l1 = args.lambda_l1
	lambda_l2 = args.lambda_l2
	node_feature_type = args.node_feature_type
	loss_type = args.loss_type
	sh_margin = args.sh_margin
	structure_neighboring_size = 1000
	khop = 2
	if(loss_type == 'AP'):
		specific_name = 'margin' + str(sh_margin) + '_'
	else:
		specific_name = ''
	if(retrain_boolean == 'True'):
		deepsea_concatenation = torch.load(model_input_path)
	else:
		deepsea_concatenation = DeepSEA_concatenation()
	if(warmstart_boolean == 'True'):
		deepsea = torch.load('../model_assessment/sota_model/deepsea_published.pkl')
		deepsea_dict = deepsea.state_dict()
		deepsea_concatenation_dict = deepsea_concatenation.state_dict()
		pretrained_layer_list = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc1.weight', 'fc1.bias']
		pretrained_dict = {k: v for k, v in deepsea_dict.items() if k in pretrained_layer_list}
		deepsea_concatenation_dict.update(pretrained_dict)
		deepsea_concatenation.load_state_dict(deepsea_concatenation_dict)
	deepsea_concatenation.cuda()
	general_path = '../graph_data/concatenation_input/'
	structure_matching_path = general_path + 'end_to_end_concatenation_input_' + resolution + '/'
	seq_label_path = general_path + 'end_to_end_concatenation_input_seq_label/'
	training_section_chr = np.load(structure_matching_path + 'training_section_chr.npy')
	training_section_index = np.load(structure_matching_path + 'training_section_index.npy')
	structure_all = np.load(general_path + '../if_matrix_' + resolution + '/' + structure_name + '_novc_whole_normalized.npy')
	if(node_feature_type == 'dnabert'):
		node_feature = np.load('dnabert_embedded_mean.npy')
	elif(node_feature_type == 'allones'):
		node_feature = np.zeros((structure_all.shape[0], 768)) + 1
	print('training data loading finished')
	num_epoch = 40
	if(loss_type == 'BCE'):
		best_valid_loss = 10
	elif(loss_type == 'AP'):
		best_valid_loss = 0
	validation_loss_list = []
	patient_count = 0
	valid_seq_x = torch.from_numpy(np.load(seq_label_path + 'seq_validation.npy')).float()
	valid_structure_matching_index = np.asarray(np.load(structure_matching_path + 'validation_matching_index.npy'), dtype = int)
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
	valid_dataloader = DataLoader(TensorDataset(valid_seq_x, valid_node_selection_index, valid_y), batch_size = 200, shuffle = False, drop_last=False)
	valid_prediction = np.zeros(valid_y.shape)
	print('validation data loading finished')
	section_size = len(training_section_chr)
	section_index = np.arange(section_size)
	validation_per_number_of_section = 20
	valid_size = valid_y.shape[0]
	num_output_channel = valid_y.shape[1]
	ap = np.zeros(num_output_channel)
	if(loss_type == 'BCE'):
		optimizer = torch.optim.Adam(deepsea_concatenation.parameters(), lr=lr, weight_decay = float(lambda_l2))
		loss_function = nn.BCELoss()
	elif(loss_type != 'AP'):
		print('unrecognized loss')
		exit()
	print('optimization started')
	print(torch.cuda.get_device_properties(0).total_memory)
	print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_cached', torch.cuda.memory_cached() / 1e9, 'memory_reserved', torch.cuda.memory_reserved() / 1e9)
	for epoch_i in range(num_epoch):
		train_losses = []
		np.random.shuffle(section_index)
		#for section_i in range(section_size):
		for section_i in range(100):
			print(section_i)
			deepsea_concatenation.train()
			train_seq_x = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_seq_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			structure_input_matching_index = torch.from_numpy(np.load(structure_matching_path + training_section_chr[section_index[section_i]] + '_section' + str(int(training_section_index[section_index[section_i]])) + '_matching_index.npy')).float()
			train_y = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_label_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			train_dataloader = DataLoader(TensorDataset(train_seq_x, structure_input_matching_index, train_y), batch_size = train_batch_size, shuffle = True, drop_last = False)
			if(loss_type == 'AP'):
				optimizer = SOAP_ADAM(deepsea_concatenation.parameters(), lr = lr, weight_decay = float(lambda_l2))
				loss_function = APLoss_SH(margin = sh_margin, beta = 0.99, data_len = train_y.shape[0] * num_output_channel)
			for train_batch_i, (train_batch_seq_x, train_batch_structure_matching_index, train_batch_y) in enumerate(train_dataloader):
				print('batch: ' + str(train_batch_i))
				if(np.random.rand() > 0.5):
					train_batch_seq_x = torch.from_numpy(np.flip(np.flip(train_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
				if(loss_type == 'AP'):
					positive_label_index_temp = np.where(train_batch_y.cpu().detach().numpy() == 1)
					positive_label_index = np.zeros((len(positive_label_index_temp), len(positive_label_index_temp[0])))
					for row_i in range(len(positive_label_index_temp)):
						positive_label_index[row_i, :] = positive_label_index_temp[row_i]
					positive_label_index = torch.from_numpy(positive_label_index).long().cuda()
				train_batch_seq_x = train_batch_seq_x.cuda()
				train_batch_structure_matching_index = np.array(train_batch_structure_matching_index.detach().numpy(), dtype = int)
				train_batch_khop_node_boolean = khop_neighbor(structure_all, train_batch_structure_matching_index, khop)
				train_batch_khop_node_index = np.where(train_batch_khop_node_boolean)[0]
				train_batch_node_selection_index = np.zeros((len(train_batch_structure_matching_index), len(train_batch_khop_node_index))) 
				for sample_i in range(len(train_batch_structure_matching_index)):
					train_batch_node_selection_index[sample_i, np.where(train_batch_khop_node_index == train_batch_structure_matching_index[sample_i])[0][0]] = 1
				train_batch_node_selection_index = torch.from_numpy(train_batch_node_selection_index).float().cuda()
				train_batch_y = train_batch_y.cuda()
				train_batch_structure_input = structure_all[train_batch_khop_node_boolean][:, train_batch_khop_node_boolean]
				train_batch_structure_input = torch.from_numpy(train_batch_structure_input).float().cuda()
				train_batch_node_input = torch.from_numpy(node_feature[train_batch_khop_node_boolean]).float().cuda()
				optimizer.zero_grad()
				print('before giving to model')
				print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_cached', torch.cuda.memory_cached() / 1e9, 'memory_reserved', torch.cuda.memory_reserved() / 1e9)
				torch.cuda.empty_cache()
				print('after cache clearning')
				print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_cached', torch.cuda.memory_cached() / 1e9, 'memory_reserved', torch.cuda.memory_reserved() / 1e9)
				out = deepsea_concatenation(train_batch_seq_x, train_batch_node_input, train_batch_structure_input, train_batch_node_selection_index)
				print('after feeding to model')
				print('memory_allocated', torch.cuda.memory_allocated() / 1e9, 'memory_cached', torch.cuda.memory_cached() / 1e9, 'memory_reserved', torch.cuda.memory_reserved() / 1e9)
				if(loss_type == 'BCE'):
					loss = loss_function(out, train_batch_y)
				elif(loss_type == 'AP'):
					batch_size_temp = train_batch_y.shape[0]
					loss = loss_function(out, train_batch_y, index_s = positive_label_index)
				#l1_penalty = deepsea_concatenation.fc1.weight.abs().sum()
				#l1_penalty += deepsea_concatenation.fc1.bias.abs().sum()
				#l1_penalty += deepsea_concatenation.fc2.weight.abs().sum()
				#l1_penalty += deepsea_concatenation.fc2.bias.abs().sum()
				#l1_penalty += deepsea_concatenation.gat1_0.W.abs().sum() + deepsea_concatenation.gat1_1.W.abs().sum() + deepsea_concatenation.gat1_2.W.abs().sum() + deepsea_concatenation.gat1_3.W.abs().sum()
				#l1_penalty += deepsea_concatenation.gat1_0.a.abs().sum() + deepsea_concatenation.gat1_1.a.abs().sum() + deepsea_concatenation.gat1_2.a.abs().sum() + deepsea_concatenation.gat1_3.a.abs().sum()
				#l1_penalty += deepsea_concatenation.gat2_0.W.abs().sum() + deepsea_concatenation.gat2_1.W.abs().sum() + deepsea_concatenation.gat2_2.W.abs().sum() + deepsea_concatenation.gat2_3.W.abs().sum()
				#l1_penalty += deepsea_concatenation.gat2_0.a.abs().sum() + deepsea_concatenation.gat2_1.a.abs().sum() + deepsea_concatenation.gat2_2.a.abs().sum() + deepsea_concatenation.gat2_3.a.abs().sum()
				#l1_penalty += deepsea_concatenation.gat3.W.abs().sum() + deepsea_concatenation.gat3.a.abs().sum()
				#loss = loss + l1_penalty * float(lambda_l1) * train_batch_y.shape[0]
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			if(section_i % validation_per_number_of_section == 0):
				deepsea_concatenation.eval()
				valid_sample_index = 0
				valid_losses = []
				for valid_batch_i, (valid_batch_seq_x, valid_batch_node_selection_index, valid_batch_y) in enumerate(valid_dataloader):
					print('valid batch: ' + str(valid_batch_i))
					valid_batch_size = valid_batch_y.shape[0]
					if(loss_type == 'AP'):
						positive_label_index_temp = np.where(valid_batch_y.cpu().detach().numpy() == 1)
						positive_label_index = np.zeros((len(positive_label_index_temp), len(positive_label_index_temp[0])))
						for row_i in range(len(positive_label_index_temp)):
							positive_label_index[row_i, :] = positive_label_index_temp[row_i]
						positive_label_index = torch.from_numpy(positive_label_index).long().cuda()
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_batch_node_selection_index = valid_batch_node_selection_index.cuda()
					valid_batch_y = valid_batch_y.cuda()
					valid_pred_half = deepsea_concatenation(valid_batch_seq_x, valid_node_feature, valid_structure_input, valid_batch_node_selection_index)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_pred_half.cpu().detach().numpy()
					if(loss_type == 'BCE'):
						val_loss = loss_function(valid_pred_half, valid_batch_y)
						valid_losses.append(val_loss.item())
					valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_pred_half = deepsea_concatenation(valid_batch_seq_x, valid_node_feature, valid_structure_input, valid_batch_node_selection_index)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] + valid_pred_half.cpu().detach().numpy()
					if(loss_type == 'BCE'):
						val_loss = loss_function(valid_pred_half, valid_batch_y)
						valid_losses.append(val_loss.item())
					valid_sample_index = valid_sample_index + valid_batch_size
				train_loss = np.average(train_losses)
				if(loss_type == 'BCE'):
					valid_loss = np.average(valid_losses)
				elif(loss_type == 'AP'):
					valid_prediction = valid_prediction / 2
					for channel_i in range(num_output_channel):
						ap[channel_i] = get_auprc(valid_prediction[:, channel_i], valid_y.numpy()[:, channel_i])
					valid_loss = np.nanmean(ap)
				if(((valid_loss < best_valid_loss) & (loss_type == 'BCE')) | ((valid_loss > best_valid_loss) & (loss_type == 'AP'))) :
					best_valid_loss = valid_loss
					torch.save(deepsea_concatenation, model_output_path + 'deepsea_gat_' + node_feature_type + '_{loss_name}_'.format(loss_name = loss_type.lower()) + specific_name + '{structure_name}_resolution{resolution}_lr{lr}_l1reg{l1reg}_l2reg{l2reg}_best.pkl'.format(structure_name = structure_name, resolution = resolution, lr = lr, l1reg = lambda_l1, l2reg = lambda_l2))
					patient_count = 0
					torch.save(deepsea_concatenation, model_output_path + 'deepsea_gat_' + node_feature_type + '_{loss_name}_'.format(loss_name = loss_type.lower()) + specific_name + '{structure_name}_resolution{resolution}_lr{lr}_l1reg{l1reg}_l2reg{l2reg}_epoch{epoch}_trainloss{trainloss}_validloss{validloss}.pkl'.format(structure_name = structure_name, resolution = resolution, lr = lr, l1reg = lambda_l1, l2reg = lambda_l2, trainloss = round(train_loss, 5), validloss = round(valid_loss, 5), epoch = round(epoch_i + section_i / section_size, 3)))
				else:
					patient_count = patient_count + 1
				if(patient_count > 40):
					exit()
if __name__=='__main__':
	main()


