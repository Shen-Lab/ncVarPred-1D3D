import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_general import DeepSEA
from model_resolution100000 import DeepSEA_concatenation
from model_diff_pathogenic import finetune_pathogenic
import argparse
from torch.utils.data import TensorDataset, DataLoader

class pathogenic_prediction_model(nn.Module):
	def __init__(self, DeepSEA_concatenation, finetune_pathogenic):
		super(pathogenic_prediction_model, self).__init__()
		self.DeepSEA_concatenation = DeepSEA_concatenation
		self.finetune_pathogenic = finetune_pathogenic
		self.finetune_pathogenic.finetune_fc1.weight.data.fill_(1.0)
		self.finetune_pathogenic.finetune_fc1.bias.data.fill_(0.0)
	def forward(self, seq_wt_input, seq_mt_input, adj_input):
		epigenetic_embedding_wt_output = self.DeepSEA_concatenation(seq_wt_input, adj_input)
		epigenetic_embedding_mt_output = self.DeepSEA_concatenation(seq_mt_input, adj_input)
		pathogenic_output = self.finetune_pathogenic(epigenetic_embedding_wt_output, epigenetic_embedding_mt_output)
		return pathogenic_output

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'concatenation architecture tunning')
	parser.add_argument('--structure_name', type = str)
	parser.add_argument('--resolution', type = str)
	parser.add_argument('--learning_rate', type = float)
	parser.add_argument('--retrain_boolean', type = str)
	parser.add_argument('--model_input_path', type = str, default = '')
	parser.add_argument('--model_output_path', type = str)
	parser.add_argument('--epigenetic_embedding_model_input_path', type = str, default = '')
	parser.add_argument('--few_shot_size', type = int, default = 100)
	parser.add_argument('--lambda_l2', type = str, default = '0')
	args = parser.parse_args()
	return args

def main():
	print('training start')
	args = parse_arguments()
	structure_name = args.structure_name
	resolution = args.resolution
	lr = args.learning_rate
	retrain_boolean = args.retrain_boolean
	model_input_path = args.model_input_path
	model_output_path = args.model_output_path
	epigenetic_embedding_model_input_path = args.epigenetic_embedding_model_input_path
	few_shot_size = args.few_shot_size
	lambda_l2 = args.lambda_l2
	if(retrain_boolean == 'True'):
		pathogenic_prediction = torch.load(model_input_path)
	else:	
		epigenetic_prediction_model = torch.load(epigenetic_embedding_model_input_path)
		finetune_pathogenic_model = finetune_pathogenic()
		pathogenic_prediction = pathogenic_prediction_model(epigenetic_prediction_model, finetune_pathogenic_model)
	pathogenic_prediction.cuda()
	general_path = '../snp_eqtl_dataset/ncVarDB/data/ncvar_'
	structure_all = np.load('../graph_data/if_matrix_' + resolution + '/' + structure_name + '_novc_whole_normalized.npy')
	train_path = general_path + 'train_'
	train_seq_wt = torch.from_numpy(np.swapaxes(np.load(train_path + 'wt_seq.npy'), 1, 2)[0:few_shot_size]).float()
	train_seq_mt = torch.from_numpy(np.swapaxes(np.load(train_path + 'mt_seq.npy'), 1, 2)[0:few_shot_size]).float()
	train_structure = torch.from_numpy(structure_all[np.load(train_path + 'structure_matching.npy')[0:few_shot_size]]).float()
	train_label_raw = np.load(train_path + 'label.npy')[0:few_shot_size]
	train_label = torch.from_numpy(train_label_raw.reshape((-1, 1))).float()
	train_dataloader = DataLoader(TensorDataset(train_seq_wt, train_seq_mt, train_structure, train_label), batch_size = 80, shuffle = False, drop_last = False)
	valid_path = general_path + 'valid_'
	valid_seq_wt = torch.from_numpy(np.swapaxes(np.load(valid_path + 'wt_seq.npy'), 1, 2)).float()
	valid_seq_mt = torch.from_numpy(np.swapaxes(np.load(valid_path + 'mt_seq.npy'), 1, 2)).float()
	valid_structure = torch.from_numpy(structure_all[np.load(valid_path + 'structure_matching.npy')]).float()
	valid_label_raw = np.load(valid_path + 'label.npy')
	valid_label = torch.from_numpy(valid_label_raw.reshape((-1, 1))).float()
	valid_dataloader = DataLoader(TensorDataset(valid_seq_wt, valid_seq_mt, valid_structure, valid_label), batch_size = 20, shuffle = False, drop_last = False)
	print('data loading finished')
	num_epoch = 1
	best_valid_loss = 100
	patient_count = 0
	optimizer = torch.optim.Adam(pathogenic_prediction.parameters(), lr=lr, weight_decay = float(lambda_l2))
	train_class_weight = train_label_raw * (1 - np.mean(train_label_raw)) + (1 - train_label_raw) * np.mean(train_label_raw)
	train_class_weight = torch.from_numpy(train_class_weight.reshape((-1, 1))).float()
	train_class_weight = train_class_weight.cuda()
	train_loss_function = nn.BCELoss(weight = train_class_weight)
	valid_class_weight = valid_label_raw * (1 - np.mean(train_label_raw)) + (1 - valid_label_raw) * np.mean(train_label_raw)
	valid_class_weight = torch.from_numpy(valid_class_weight.reshape((-1, 1))).float()
	valid_class_weight = valid_class_weight.cuda()
	valid_loss_function = nn.BCELoss(weight = valid_class_weight)
	print('optimization started')
	for epoch_i in range(num_epoch):
		train_losses = []
		pathogenic_prediction.train()
		for train_batch_i, (train_batch_seq_wt, train_batch_seq_mt, train_batch_structure_x, train_batch_y) in enumerate(train_dataloader):
			if(np.random.rand() > 0.5):
				train_batch_seq_wt = torch.from_numpy(np.flip(np.flip(train_batch_seq_wt.detach().numpy(), 1), 2).copy()).float()
				train_batch_seq_mt = torch.from_numpy(np.flip(np.flip(train_batch_seq_mt.detach().numpy(), 1), 2).copy()).float()
			train_batch_seq_wt = train_batch_seq_wt.cuda()
			train_batch_seq_mt = train_batch_seq_mt.cuda()
			train_batch_structure_x = train_batch_structure_x.cuda()
			train_batch_y = train_batch_y.cuda()
			optimizer.zero_grad()
			out = pathogenic_prediction(train_batch_seq_wt, train_batch_seq_mt, train_batch_structure_x)
			loss = train_loss_function(out, train_batch_y)
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())
		pathogenic_prediction.eval()
		valid_sample_index = 0
		valid_loss = 0
		for valid_batch_i, (valid_batch_seq_wt, valid_batch_seq_mt, valid_batch_structure_x, valid_batch_y) in enumerate(valid_dataloader):
			valid_batch_size_i = valid_batch_seq_wt.shape[0]
			valid_batch_seq_wt = valid_batch_seq_wt.cuda()
			valid_batch_seq_mt = valid_batch_seq_mt.cuda()
			valid_batch_structure_x = valid_batch_structure_x.cuda()
			valid_batch_y = valid_batch_y.cuda()
			valid_pred_firsthalf = pathogenic_prediction(valid_batch_seq_wt, valid_batch_seq_mt, valid_batch_structure_x)
			valid_loss += valid_loss_function(valid_pred_firsthalf, valid_batch_y).item() * valid_batch_size_i
			valid_batch_seq_wt = torch.from_numpy(np.flip(np.flip(valid_batch_seq_wt.cpu().detach().numpy(), 1), 2).copy()).float()
			valid_batch_seq_mt = torch.from_numpy(np.flip(np.flip(valid_batch_seq_mt.cpu().detach().numpy(), 1), 2).copy()).float()
			valid_batch_seq_wt = valid_batch_seq_wt.cuda()
			valid_batch_seq_mt = valid_batch_seq_mt.cuda()
			valid_pred_secondhalf = pathogenic_prediction(valid_batch_seq_wt, valid_batch_seq_mt, valid_batch_structure_x)
			valid_loss += valid_loss_function(valid_pred_secondhalf, valid_batch_y).item() * valid_batch_size_i
			valid_sample_index += valid_batch_size_i
		train_loss = np.average(train_losses)
		valid_loss = valid_loss / valid_sample_index / 2
		if(valid_loss < best_valid_loss):
			best_valid_loss = valid_loss
			print(best_valid_loss)
			patient_count = 0
			torch.save(pathogenic_prediction, model_output_path + 'deepsea_fc_diff_pathogenic_{structure_name}_resolution{resolution}_lr{lr}_fewshotsize{few_shot_size}_l2reg{l2reg}_epoch{epoch}_trainloss{trainloss}_validloss{validloss}.pkl'.format(structure_name = structure_name, resolution = resolution, lr = lr, few_shot_size = str(int(few_shot_size)), l2reg = lambda_l2, trainloss = round(train_loss, 5), validloss = round(valid_loss, 5), epoch = epoch_i))
		else:
			patient_count = patient_count + 1
		if(patient_count > 40):
			exit()

if __name__=='__main__':
	main()


