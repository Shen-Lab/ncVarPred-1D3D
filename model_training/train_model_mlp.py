import numpy as np
from model_seq_only import *
from model_mlp import *
import argparse
from torch.utils.data import TensorDataset, DataLoader

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'concatenation architecture tunning')
	parser.add_argument('--structure_name', type = str)
	parser.add_argument('--learning_rate', type = float)
	parser.add_argument('--model_version', type = str)
	parser.add_argument('--warmstart_boolean', type = str)
	parser.add_argument('--warmstart_model_path', type = str, default = '')
	parser.add_argument('--retrain_boolean', type = str)
	parser.add_argument('--model_input_path', type = str, default = '')
	parser.add_argument('--model_output_path', type = str)
	parser.add_argument('--lambda_l1', type = str, default = '0')
	parser.add_argument('--lambda_l2', type = str, default = '0')
	args = parser.parse_args()
	return args

def main():
	print('training start')
	args = parse_arguments()
	structure_name = args.structure_name
	lr = args.learning_rate
	model_version = args.model_version
	warmstart_boolean = args.warmstart_boolean
	warmstart_model_path = args.warmstart_model_path
	retrain_boolean = args.retrain_boolean
	model_input_path = args.model_input_path
	model_output_path = args.model_output_path
	lambda_l1 = args.lambda_l1
	lambda_l2 = args.lambda_l2
	if(retrain_boolean == 'True'):
		model = torch.load(model_input_path)
	else:
		if(model_version == 'CNN_MLP'):
			model = DeepSEA_concatenation()
		elif(model_version == 'CNN_RNN_MLP'):
			model = DanQ_concatenation()
		elif(model_version == 'CNN'):
			model = DeepSEA()
		elif(model_version == 'CNN_RNN'):
			model = DanQ()
	if(warmstart_boolean == 'True'):
		pretrained_model = torch.load(warmstart_model_path)
		pretrained_model_dict = pretrained_model.state_dict()
		model_dict = model.state_dict()
		if(model_version == 'CNN_MLP'):
			pretrained_layer_list = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc1.weight', 'fc1.bias']
		elif(model_version == 'CNN_RNN_MLP'):
			pretrained_layer_list = ['conv1.weight', 'conv1.bias', 'bilstm.weight_hh_l0', 'bilstm.weight_hh_l0_reverse', 'bilstm.weight_ih_l0', 'bilstm.weight_ih_l0_reverse', 'bias_hh_l0', 'bias_hh_l0_reverse', 'bias_ih_l0', 'bias_ih_l0_reverse', 'fc1.weight', 'fc1.bias']
		pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in pretrained_layer_list}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	model.cuda()
	general_path = '../training_data/'
	structure_matching_path = general_path + 'deepsea_structure_matching_index/'
	seq_label_path = general_path + 'deepsea_seq_label/'
	training_section_chr = np.load(structure_matching_path + 'training_section_chr.npy')
	training_section_index = np.load(structure_matching_path + 'training_section_index.npy')
	structure_all = np.load(general_path + '/if_matrix/' + structure_name + '_novc_whole_normalized.npy')
	print('training data loading finished')
	num_epoch = 40
	best_valid_loss = 10
	validation_loss_list = []
	patient_count = 0
	valid_seq_x = torch.from_numpy(np.load(seq_label_path + 'seq_validation.npy')).float()
	valid_structure_matching_index = np.asarray(np.load(structure_matching_path + 'validation_matching_index.npy'), dtype = int)
	valid_y = torch.from_numpy(np.load(seq_label_path + 'label_validation.npy')).float()
	valid_structure_x = torch.from_numpy(structure_all[valid_structure_matching_index, :]).float()
	valid_dataloader = DataLoader(TensorDataset(valid_seq_x, valid_structure_x, valid_y), batch_size = 512, shuffle = False, drop_last=False)
	valid_prediction = np.zeros(valid_y.shape)
	print('validation data loading finished')
	section_size = len(training_section_chr)
	section_index = np.arange(section_size)
	validation_per_number_of_section = 20
	valid_size = valid_y.shape[0]
	num_output_channel = valid_y.shape[1]
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = float(lambda_l2))
	loss_function = nn.BCELoss()
	print('optimization started')
	for epoch_i in range(num_epoch):
		train_losses = []
		np.random.shuffle(section_index)
		for section_i in range(section_size):
			model.train()
			train_seq_x = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_seq_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			structure_input_matching_index = np.load(structure_matching_path + training_section_chr[section_index[section_i]] + '_section' + str(int(training_section_index[section_index[section_i]])) + '_matching_index.npy')
			train_structure_x = torch.from_numpy(structure_all[np.asarray(structure_input_matching_index, dtype = int), :]).float()
			train_y = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_label_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			train_dataloader = DataLoader(TensorDataset(train_seq_x, train_structure_x, train_y), batch_size = 512, shuffle = True, drop_last = False)
			for train_batch_i, (train_batch_seq_x, train_batch_structure_x, train_batch_y) in enumerate(train_dataloader):
				print('batch: ' + str(train_batch_i))
				if(np.random.rand() > 0.5):
					train_batch_seq_x = torch.from_numpy(np.flip(np.flip(train_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
				positive_label_index = train_batch_y.long().cuda()
				train_batch_seq_x = train_batch_seq_x.cuda()
				train_batch_structure_x = train_batch_structure_x.cuda()
				train_batch_y = train_batch_y.cuda()
				optimizer.zero_grad()
				l1_penalty = model.fc1.weight.abs().sum()
				l1_penalty += model.fc1.bias.abs().sum()
				l1_penalty += model.fc2.weight.abs().sum()
				l1_penalty += model.fc2.bias.abs().sum()
				if(model_version in ['CNN', 'CNN_RNN']):
					out = model(train_batch_seq_x)
				elif(model_version in ['CNN_MLP', 'CNN_RNN_MLP']):
					out = model(train_batch_seq_x, train_batch_structure_x)	
					l1_penalty += model.structure_fc1.weight.abs().sum()
					l1_penalty += model.structure_fc1.bias.abs().sum()
					l1_penalty += model.structure_fc2.weight.abs().sum()
					l1_penalty += model.structure_fc2.bias.abs().sum()
					l1_penalty += model.structure_fc3.weight.abs().sum()
					l1_penalty += model.structure_fc3.bias.abs().sum()
				loss = loss_function(out, train_batch_y)
				loss = loss + l1_penalty * float(lambda_l1) * train_batch_y.shape[0]
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			if(section_i % validation_per_number_of_section == 0):
				model.eval()
				valid_sample_index = 0
				valid_losses = []
				for valid_batch_i, (valid_batch_seq_x, valid_batch_structure_x, valid_batch_y) in enumerate(valid_dataloader):
					valid_batch_size = valid_batch_y.shape[0]
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_batch_structure_x = valid_batch_structure_x.cuda()
					valid_batch_y = valid_batch_y.cuda()
					if(model_version in ['CNN', 'CNN_RNN']):
						valid_pred_half = model(valid_batch_seq_x)
					elif(model_version in ['CNN_MLP', 'CNN_RNN_MLP']):
						valid_pred_half = model(valid_batch_seq_x, valid_batch_structure_x)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_pred_half.cpu().detach().numpy()
					val_loss = loss_function(valid_pred_half, valid_batch_y)
					valid_losses.append(val_loss.item())
					valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					if(model_version in ['CNN', 'CNN_RNN']):
						valid_pred_half = model(valid_batch_seq_x)
					elif(model_version in ['CNN_MLP', 'CNN_RNN_MLP']):
						valid_pred_half = model(valid_batch_seq_x, valid_batch_structure_x)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] + valid_pred_half.cpu().detach().numpy()
					val_loss = loss_function(valid_pred_half, valid_batch_y)
					valid_losses.append(val_loss.item())
					valid_sample_index = valid_sample_index + valid_batch_size
				train_loss = np.average(train_losses)
				valid_loss = np.average(valid_losses)
				if(valid_loss < best_valid_loss):
					best_valid_loss = valid_loss
					torch.save(model, model_output_path + model_version + '_{structure_name}_lr{lr}_l1reg{l1reg}_l2reg{l2reg}_best.pkl'.format(structure_name = structure_name, lr = lr, l1reg = lambda_l1, l2reg = lambda_l2))
					patient_count = 0
					torch.save(model, model_output_path + model_version + '_{structure_name}_lr{lr}_l1reg{l1reg}_l2reg{l2reg}_epoch{epoch}_trainloss{trainloss}_validloss{validloss}.pkl'.format(structure_name = structure_name, lr = lr, l1reg = lambda_l1, l2reg = lambda_l2, trainloss = round(train_loss, 5), validloss = round(valid_loss, 5), epoch = round(epoch_i + section_i / section_size, 3)))
				else:
					patient_count = patient_count + 1
				if(patient_count > 40):
					exit()
if __name__=='__main__':
	main()


