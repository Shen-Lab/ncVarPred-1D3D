import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_general import DeepSEA
from model_resolution100000 import DeepSEA_concatenation
from model_diff_pathogenic import finetune_pathogenic
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

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

def get_auroc(preds, obs):
	fpr, tpr, thresholds  = metrics.roc_curve(obs, preds, drop_intermediate=False)
	auroc = metrics.auc(fpr, tpr)
	return auroc

def get_auprc(preds, obs):
	precision, recall, thresholds  = metrics.precision_recall_curve(obs, preds)
	auprc = metrics.auc(recall, precision)
	return auprc

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'concatenation architecture tunning')
	parser.add_argument('--structure_name', type = str)
	parser.add_argument('--resolution', type = str)
	parser.add_argument('--model_input_path', type = str, default = '')
	parser.add_argument('--result_output_path', type = str)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	structure_name = args.structure_name
	resolution = args.resolution
	model_input_path = args.model_input_path
	result_output_path = args.result_output_path
	pathogenic_prediction = torch.load(model_input_path)
	pathogenic_prediction.cuda()
	general_path = '../snp_eqtl_dataset/ncVarDB/data/ncvar_'
	structure_all = np.load('../graph_data/if_matrix_' + resolution + '/' + structure_name + '_novc_whole_normalized.npy')
	test_path = general_path + 'test_'
	test_seq_wt = torch.from_numpy(np.swapaxes(np.load(test_path + 'wt_seq.npy'), 1, 2)).float()
	test_seq_mt = torch.from_numpy(np.swapaxes(np.load(test_path + 'mt_seq.npy'), 1, 2)).float()
	test_structure = torch.from_numpy(structure_all[np.load(test_path + 'structure_matching.npy')]).float()
	test_label_raw = np.load(test_path + 'label.npy')
	test_label = torch.from_numpy(test_label_raw.reshape((-1, 1))).float()
	test_batch_size = 128
	test_dataloader = DataLoader(TensorDataset(test_seq_wt, test_seq_mt, test_structure, test_label), batch_size = test_batch_size, shuffle = False, drop_last = False)
	test_pred = np.zeros(len(test_label_raw))
	print('data loading finished')
	for test_batch_i, (test_batch_seq_wt, test_batch_seq_mt, test_batch_structure_x, test_batch_y) in enumerate(test_dataloader):
		test_batch_seq_wt = test_batch_seq_wt.cuda()
		test_batch_seq_mt = test_batch_seq_mt.cuda()
		test_batch_structure_x = test_batch_structure_x.cuda()
		test_batch_pred_firsthalf = pathogenic_prediction(test_batch_seq_wt, test_batch_seq_mt, test_batch_structure_x)
		test_batch_seq_wt = torch.from_numpy(np.flip(np.flip(test_batch_seq_wt.cpu().detach().numpy(), 1), 2).copy()).float()
		test_batch_seq_mt = torch.from_numpy(np.flip(np.flip(test_batch_seq_mt.cpu().detach().numpy(), 1), 2).copy()).float()
		test_batch_seq_wt = test_batch_seq_wt.cuda()
		test_batch_seq_mt = test_batch_seq_mt.cuda()
		test_batch_pred_secondhalf = pathogenic_prediction(test_batch_seq_wt, test_batch_seq_mt, test_batch_structure_x)
		test_batch_pred = (test_batch_pred_firsthalf.cpu().detach().numpy() + test_batch_pred_secondhalf.cpu().detach().numpy()) / 2
		test_pred[int(test_batch_size * test_batch_i):int(test_batch_size * test_batch_i + test_batch_size)] = test_batch_pred.flatten()
	test_auroc = get_auroc(test_pred, test_label_raw)
	test_auprc = get_auprc(test_pred, test_label_raw)
	print(model_input_path)
	print(str(test_auroc) + ' & ' + str(test_auprc))
	np.save(result_output_path, np.array([test_auroc, test_auprc]))
	np.save(result_output_path.split('_auroc_auprc')[0] + '_pred.npy', test_pred)

if __name__=='__main__':
	main()


