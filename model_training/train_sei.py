import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sei import Sei

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'concatenation architecture tunning')
	parser.add_argument('--learning_rate', type = float)
	parser.add_argument('--retrain_boolean', type = str)
	parser.add_argument('--l2reg', type = float)
	parser.add_argument('--model_input_path', type = str, default = '')
	parser.add_argument('--model_output_path', type = str)
	args = parser.parse_args()
	return args

def main():
	print('training start')
	args = parse_arguments()
	lr = args.learning_rate
	retrain_boolean = args.retrain_boolean
	l2reg = args.l2reg
	model_input_path = args.model_input_path
	model_output_path = args.model_output_path
	if(retrain_boolean == 'True'):
		model = torch.load(model_input_path)
	else:
		model = Sei()
	model.cuda()
	general_path = '../training_data/'
	seq_label_path = general_path + 'sei_seq_label/'
	structure_matching_path = general_path + 'sei_structure_matching_index/'
	training_section_chr = np.load(structure_matching_path + 'training_section_chr.npy')
	training_section_index = np.load(structure_matching_path + 'training_section_index.npy')
	print('training data loading finished')
	num_epoch = 40
	best_valid_loss = 10
	validation_loss_list = []
	patient_count = 0
	valid_seq_x = torch.from_numpy(np.load(seq_label_path + 'seq_validation.npy')).float()
	valid_y = torch.from_numpy(np.load(seq_label_path + 'label_validation.npy')).float()
	valid_dataloader = DataLoader(TensorDataset(valid_seq_x, valid_y), batch_size = 32, shuffle = False, drop_last=False)
	valid_prediction = np.zeros(valid_y.shape)
	print('validation data loading finished')
	section_size = len(training_section_chr)
	section_index = np.arange(section_size)
	validation_per_number_of_section = 20
	valid_size = valid_y.shape[0]
	num_output_channel = valid_y.shape[1]
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = l2reg)
	loss_function = nn.BCELoss()
	print('optimization started')
	for epoch_i in range(num_epoch):
		train_losses = []
		np.random.shuffle(section_index)
		for section_i in range(section_size):
			print(section_i)
			model.train()
			train_seq_x = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_seq_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			train_y = torch.from_numpy(np.load(seq_label_path + training_section_chr[section_index[section_i]] + '_label_section' + str(int(training_section_index[section_index[section_i]])) + '.npy')).float()
			train_dataloader = DataLoader(TensorDataset(train_seq_x, train_y), batch_size = 32, shuffle = True, drop_last = False)
			for train_batch_i, (train_batch_seq_x, train_batch_y) in enumerate(train_dataloader):
				print('batch: ' + str(train_batch_i))
				if(np.random.rand() > 0.5):
					train_batch_seq_x = torch.from_numpy(np.flip(np.flip(train_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
				train_batch_seq_x = train_batch_seq_x.cuda()
				train_batch_y = train_batch_y.cuda()
				optimizer.zero_grad()
				out = model(train_batch_seq_x)
				loss = loss_function(out, train_batch_y)
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			if(section_i % validation_per_number_of_section == 0):
				model.eval()
				valid_sample_index = 0
				valid_losses = []
				for valid_batch_i, (valid_batch_seq_x, valid_batch_y) in enumerate(valid_dataloader):
					print('valid batch: ' + str(valid_batch_i))
					valid_batch_size = valid_batch_y.shape[0]
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_batch_y = valid_batch_y.cuda()
					valid_pred_half = model(valid_batch_seq_x)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_pred_half.cpu().detach().numpy()
					val_loss = loss_function(valid_pred_half, valid_batch_y)
					valid_losses.append(val_loss.item())
					valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
					valid_batch_seq_x = valid_batch_seq_x.cuda()
					valid_pred_half = model(valid_batch_seq_x)
					valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] = valid_prediction[int(valid_sample_index):int(valid_sample_index + valid_batch_size)] + valid_pred_half.cpu().detach().numpy()
					val_loss = loss_function(valid_pred_half, valid_batch_y)
					valid_losses.append(val_loss.item())
					valid_sample_index = valid_sample_index + valid_batch_size
				train_loss = np.average(train_losses)
				valid_loss = np.average(valid_losses)
				if(valid_loss < best_valid_loss) :
					best_valid_loss = valid_loss
					torch.save(model, model_output_path + 'Sei_lr{lr}_l2reg{l2reg}_best.pkl'.format(lr = lr, l2reg = l2reg))
					patient_count = 0
					torch.save(model, model_output_path + 'Sei_lr{lr}_l2reg{l2reg}_epoch{epoch}_trainloss{trainloss}_validloss{validloss}.pkl'.format(lr = lr, l2reg = l2reg, trainloss = round(train_loss, 5), validloss = round(valid_loss, 5), epoch = round(epoch_i + section_i / section_size, 3)))
				else:
					patient_count = patient_count + 1
				if(patient_count > 40):
					exit()
if __name__=='__main__':
	main()


