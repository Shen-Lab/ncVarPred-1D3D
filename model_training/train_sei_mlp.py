import numpy as np
from layers import GraphConvolution
from sei_mlp import Sei_concatenation
import argparse
from torch.utils.data import TensorDataset, DataLoader


def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'concatenation architecture tunning')
	parser.add_argument('--learning_rate', type = float)
	parser.add_argument('--warmstart_boolean', type = str)
	parser.add_argument('--retrain_boolean', type = str)
	parser.add_argument('--structure_name', type = str)
	parser.add_argument('--model_input_path', type = str, default = '')
	parser.add_argument('--model_output_path', type = str)
	parser.add_argument('--lambda_l2', type = str, default = '0')
	args = parser.parse_args()
	return args

def main():
	print('training start')
	args = parse_arguments()
	lr = args.learning_rate
	warmstart_boolean = args.warmstart_boolean
	retrain_boolean = args.retrain_boolean
	structure_name = args.structure_name
	model_input_path = args.model_input_path
	model_output_path = args.model_output_path
	lambda_l2 = args.lambda_l2
	if(retrain_boolean == 'True'):
		model = torch.load(model_input_path)
	else:
		model = Sei_concatenation()
	if(warmstart_boolean == 'True'):
		pretrained_model = torch.load('../trained_model/SOTA/Sei_reproduced.pkl')
		pretrained_model_dict = pretrained_model.state_dict()
		model_dict = model.state_dict()
		pretrained_layer_list = ['lconv1.0.weight', 'lconv1.0.bias', 'lconv1.1.weight', 'lconv1.1.bias', 'conv1.0.weight', 'conv1.0.bias', 'conv1.2.weight', 'conv1.2.bias', 'lconv2.2.weight', 'lconv2.2.bias', 'lconv2.3.weight', 'lconv2.3.bias', 'conv2.1.weight', 'conv2.1.bias', 'conv2.3.weight', 'conv2.3.bias', 'lconv3.2.weight', 'lconv3.2.bias', 'lconv3.3.weight', 'lconv3.3.bias', 'conv3.1.weight', 'conv3.1.bias', 'conv3.3.weight', 'conv3.3.bias', 'dconv1.1.weight', 'dconv1.1.bias', 'dconv2.1.weight', 'dconv2.1.bias', 'dconv3.1.weight', 'dconv3.1.bias', 'dconv4.1.weight', 'dconv4.1.bias', 'dconv5.1.weight', 'dconv5.1.bias']
		pretrained_dict = {k: v for k, v in pretrained_model_dict.items() if k in pretrained_layer_list}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	model.cuda()
	general_path = '../training_data/'
	seq_label_path = general_path + 'sei_seq_label/'
	structure_matching_path = general_path + 'sei_structure_matching_index/'
	training_section_chr = np.load(structure_matching_path + 'training_section_chr.npy')
	training_section_index = np.load(structure_matching_path + 'training_section_index.npy')
	structure_all = np.load(general_path + 'if_matrix/' + structure_name + '_novc_whole_normalized.npy')
	print('training data loading finished')
	num_epoch = 40
	best_valid_loss = 10
	validation_loss_list = []
	patient_count = 0
	valid_seq_x = torch.from_numpy(np.load(seq_label_path + 'seq_validation.npy')).float()
	valid_structure_matching_index = np.asarray(np.load(structure_matching_path + 'validation_matching_index.npy'), dtype = int)
	valid_structure_x = torch.from_numpy(structure_all[valid_structure_matching_index, :]).float()
	valid_y = torch.from_numpy(np.load(seq_label_path + 'label_validation.npy')).float()
	valid_dataloader = DataLoader(TensorDataset(valid_seq_x, valid_structure_x, valid_y), batch_size = 50, shuffle = False, drop_last=False)
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
			print(section_i)
			model.train()
			train_seq_x = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_seq_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			structure_input_matching_index = np.load(structure_matching_path + training_section_chr[section_index[section_i]] + '_section' + str(int(training_section_index[section_index[section_i]])) + '_matching_index.npy')
			train_structure_x = torch.from_numpy(structure_all[np.asarray(structure_input_matching_index, dtype = int), :]).float()
			train_y = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_label_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			train_dataloader = DataLoader(TensorDataset(train_seq_x, train_structure_x, train_y), batch_size = 64, shuffle = True, drop_last = False)
			for train_batch_i, (train_batch_seq_x, train_batch_structure_x, train_batch_y) in enumerate(train_dataloader):
				print('batch: ' + str(train_batch_i))
				if(np.random.rand() > 0.5):
					train_batch_seq_x = torch.from_numpy(np.flip(np.flip(train_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
				train_batch_seq_x = train_batch_seq_x.cuda()
				train_batch_structure_x = train_batch_structure_x.cuda()
				train_batch_y = train_batch_y.cuda()
				optimizer.zero_grad()
				out = model(train_batch_seq_x, train_batch_structure_x)
				loss = loss_function(out, train_batch_y)
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			if(section_i % validation_per_number_of_section == 0):
				model.eval()
				valid_sample_index = 0
				valid_losses = []
				for valid_batch_i, (valid_batch_seq_x, valid_batch_structure_x, valid_batch_y) in enumerate(valid_dataloader):
					print('valid batch: ' + str(valid_batch_i))
					valid_batch_size = valid_batch_y.shape[0]
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_batch_structure_x = valid_batch_structure_x.cuda()
					valid_batch_y = valid_batch_y.cuda()
					valid_pred_half = model(valid_batch_seq_x, valid_batch_structure_x)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_pred_half.cpu().detach().numpy()
					val_loss = loss_function(valid_pred_half, valid_batch_y)
					valid_losses.append(val_loss.item())
					valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_pred_half = model(valid_batch_seq_x, valid_batch_structure_x)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] + valid_pred_half.cpu().detach().numpy()
					val_loss = loss_function(valid_pred_half, valid_batch_y)
					valid_losses.append(val_loss.item())
					valid_sample_index = valid_sample_index + valid_batch_size
				train_loss = np.average(train_losses)
				valid_loss = np.average(valid_losses)
				if(valid_loss < best_valid_loss):
					best_valid_loss = valid_loss
					torch.save(model, model_output_path + 'Sei_MLP_' + structure_name + '_lr{lr}_l2reg{l2reg}_best.pkl'.format(lr = lr, l2reg = lambda_l2))
					patient_count = 0
					torch.save(model, model_output_path + 'Sei_MLP_' + structure_name + '_lr{lr}_l2reg{l2reg}_epoch{epoch}_trainloss{trainloss}_validloss{validloss}.pkl'.format(lr = lr, l2reg = lambda_l2, trainloss = round(train_loss, 5), validloss = round(valid_loss, 5), epoch = round(epoch_i + section_i / section_size, 3)))
				else:
					patient_count = patient_count + 1
				if(patient_count > 40):
					exit()
if __name__=='__main__':
	main()


