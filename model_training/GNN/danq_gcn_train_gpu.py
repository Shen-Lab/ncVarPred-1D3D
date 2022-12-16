import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from sklearn import metrics
from model_general import *
from model_gcn import DanQ_concatenation
import argparse
from torch.utils.data import TensorDataset, DataLoader
from libauc.optimizers import SOAP_SGD, SOAP_ADAM
from libauc.losses import APLoss_SH

def get_auprc(preds, obs):
	precision, recall, thresholds  = metrics.precision_recall_curve(obs, preds)
	auprc = metrics.auc(recall, precision)
	return auprc

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
	if(loss_type == 'AP'):
		specific_name = 'margin' + str(sh_margin) + '_'
	else:
		specific_name = ''
	if(retrain_boolean == 'True'):
		danq_concatenation = torch.load(model_input_path)
	else:
		danq_concatenation = DanQ_concatenation()
	if(warmstart_boolean == 'True'):
		danq = torch.load('../model_assessment/sota_model/DanQ_replicated.pkl')
		danq_dict = danq.state_dict()
		danq_concatenation_dict = danq_concatenation.state_dict()
		pretrained_layer_list = ['conv1.weight', 'conv1.bias', 'bilstm.weight_hh_l0', 'bilstm.weight_hh_l0_reverse', 'bilstm.weight_ih_l0', 'bilstm.weight_ih_l0_reverse', 'bias_hh_l0', 'bias_hh_l0_reverse', 'bias_ih_l0', 'bias_ih_l0_reverse', 'fc1.weight', 'fc1.bias']
		pretrained_dict = {k: v for k, v in danq_dict.items() if k in pretrained_layer_list}
		danq_concatenation_dict.update(pretrained_dict)
		danq_concatenation.load_state_dict(danq_concatenation_dict)
	danq_concatenation.cuda()
	general_path = '../graph_data/concatenation_input/'
	structure_matching_path = general_path + 'end_to_end_concatenation_input_' + resolution + '/'
	seq_label_path = general_path + 'end_to_end_concatenation_input_seq_label/'
	training_section_chr = np.load(structure_matching_path + 'training_section_chr.npy')
	training_section_index = np.load(structure_matching_path + 'training_section_index.npy')
	structure_all = torch.from_numpy(np.load(general_path + '../if_matrix_' + resolution + '/' + structure_name + '_novc_whole_normalized.npy')).float().cuda()
	if(node_feature_type == 'dnabert'):
		node_feature = torch.from_numpy(np.load('dnabert_embedded_mean.npy')).float().cuda()
	elif(node_feature_type == 'allones'):
		node_feature = torch.from_numpy(np.zeros((structure_all.shape[0], 768)) + 1).float().cuda()
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
	valid_y = torch.from_numpy(np.load(seq_label_path + 'label_validation.npy')).float()
	valid_node_selection_index = np.zeros((len(valid_structure_matching_index), structure_all.shape[0]))
	for sample_i in range(len(valid_structure_matching_index)):
		valid_node_selection_index[sample_i, valid_structure_matching_index[sample_i]] = 1
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
		optimizer = torch.optim.Adam(danq_concatenation.parameters(), lr=lr, weight_decay = float(lambda_l2))
		loss_function = nn.BCELoss()
	elif(loss_type != 'AP'):
		print('unrecognized loss')
		exit()
	print('optimization started')
	for epoch_i in range(num_epoch):
		train_losses = []
		np.random.shuffle(section_index)
		for section_i in range(section_size):
			print(section_i)
			danq_concatenation.train()
			train_seq_x = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_seq_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			structure_input_matching_index = torch.from_numpy(np.load(structure_matching_path + training_section_chr[section_index[section_i]] + '_section' + str(int(training_section_index[section_index[section_i]])) + '_matching_index.npy')).float()
			train_y = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_label_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			train_dataloader = DataLoader(TensorDataset(train_seq_x, structure_input_matching_index, train_y), batch_size = train_batch_size, shuffle = True, drop_last = False)
			if(loss_type == 'AP'):
				optimizer = SOAP_ADAM(danq_concatenation.parameters(), lr = lr, weight_decay = float(lambda_l2))
				loss_function = APLoss_SH(margin = sh_margin, beta = 0.99, data_len = train_y.shape[0] * num_output_channel)
			for train_batch_i, (train_batch_seq_x, train_batch_structure_matching_index, train_batch_y) in enumerate(train_dataloader):
				print('batch: ' + str(train_batch_i))
				if(np.random.rand() > 0.5):
					train_batch_seq_x = torch.from_numpy(np.flip(np.flip(train_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
				positive_label_index = train_batch_y.long().cuda()
				train_batch_seq_x = train_batch_seq_x.cuda()
				train_batch_node_selection_index = np.zeros((train_batch_seq_x.shape[0], structure_all.shape[0])) 
				train_batch_structure_matching_index = np.array(train_batch_structure_matching_index.detach().numpy(), dtype = int)
				for sample_i in range(len(train_batch_structure_matching_index)):
					train_batch_node_selection_index[sample_i, train_batch_structure_matching_index[sample_i]] = 1
				train_batch_node_selection_index = torch.from_numpy(train_batch_node_selection_index).float().cuda()
				train_batch_y = train_batch_y.cuda()
				optimizer.zero_grad()
				out = danq_concatenation(train_batch_seq_x, node_feature, structure_all, train_batch_node_selection_index)
				if(loss_type == 'BCE'):
					loss = loss_function(out, train_batch_y)
				elif(loss_type == 'AP'):
					batch_size_temp = train_batch_y.shape[0]
					loss = loss_function(out, train_batch_y, index_s = positive_label_index)
				l1_penalty = danq_concatenation.fc1.weight.abs().sum()
				l1_penalty += danq_concatenation.fc1.bias.abs().sum()
				l1_penalty += danq_concatenation.fc2.weight.abs().sum()
				l1_penalty += danq_concatenation.fc2.bias.abs().sum()
				l1_penalty += danq_concatenation.gcn1.weight.abs().sum()
				l1_penalty += danq_concatenation.gcn1.bias.abs().sum()
				l1_penalty += danq_concatenation.gcn2.weight.abs().sum()
				l1_penalty += danq_concatenation.gcn2.bias.abs().sum()
				l1_penalty += danq_concatenation.gcn3.weight.abs().sum()
				l1_penalty += danq_concatenation.gcn3.bias.abs().sum()
				loss = loss + l1_penalty * float(lambda_l1) * train_batch_y.shape[0]
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			if(section_i % validation_per_number_of_section == 0):
				danq_concatenation.eval()
				valid_sample_index = 0
				valid_losses = []
				for valid_batch_i, (valid_batch_seq_x, valid_batch_node_selection_index, valid_batch_y) in enumerate(valid_dataloader):
					print('valid batch: ' + str(valid_batch_i))
					valid_batch_size = valid_batch_y.shape[0]
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_batch_node_selection_index = valid_batch_node_selection_index.cuda()
					valid_batch_y = valid_batch_y.cuda()
					valid_pred_half = danq_concatenation(valid_batch_seq_x, node_feature, structure_all, valid_batch_node_selection_index)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_pred_half.cpu().detach().numpy()
					if(loss_type == 'BCE'):
						val_loss = loss_function(valid_pred_half, valid_batch_y)
						valid_losses.append(val_loss.item())
					valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_pred_half = danq_concatenation(valid_batch_seq_x, node_feature, structure_all, valid_batch_node_selection_index)
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
					torch.save(danq_concatenation, model_output_path + 'danq_gcn_' + node_feature_type + '_{loss_name}_'.format(loss_name = loss_type.lower()) + specific_name + '{structure_name}_resolution{resolution}_lr{lr}_l1reg{l1reg}_l2reg{l2reg}_best.pkl'.format(structure_name = structure_name, resolution = resolution, lr = lr, l1reg = lambda_l1, l2reg = lambda_l2))
					patient_count = 0
					torch.save(danq_concatenation, model_output_path + 'danq_gcn_' + node_feature_type + '_{loss_name}_'.format(loss_name = loss_type.lower()) + specific_name + '{structure_name}_resolution{resolution}_lr{lr}_l1reg{l1reg}_l2reg{l2reg}_epoch{epoch}_trainloss{trainloss}_validloss{validloss}.pkl'.format(structure_name = structure_name, resolution = resolution, lr = lr, l1reg = lambda_l1, l2reg = lambda_l2, trainloss = round(train_loss, 5), validloss = round(valid_loss, 5), epoch = round(epoch_i + section_i / section_size, 3)))
				else:
					patient_count = patient_count + 1
				if(patient_count > 40):
					exit()
if __name__=='__main__':
	main()


