import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_general import DeepSEA
from model_resolution100000 import DeepSEA_concatenation
from torch.utils.data import TensorDataset, DataLoader
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'inference')
	parser.add_argument('--our_model_name', type = str)
	parser.add_argument('--our_model_path', type = str)
	parser.add_argument('--sota_model_name', type = str)
	parser.add_argument('--sota_model_path', type = str)
	parser.add_argument('--seq_input_name', type = str)
	parser.add_argument('--seq_input_path', type = str)
	parser.add_argument('--structure_input_matching_name', type = str)
	parser.add_argument('--structure_input_path', type = str)
	parser.add_argument('--structure_input_matching_path', type = str)
	parser.add_argument('--feature_selected_boolean_path', type = str)
	parser.add_argument('--experiment_name', type = str)
	parser.add_argument('--output_path', type = str)
	parser.add_argument('--output_our_model_name', type = str)
	parser.add_argument('--output_sota_name', type = str)
	parser.add_argument('--seq_specific_name', type = str)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	our_model_name = args.our_model_name
	our_model_path = args.our_model_path
	sota_model_name = args.sota_model_name.split('.pkl')[0] + '.pkl'
	sota_model_path = args.sota_model_path
	seq_input_path = args.seq_input_path
	seq_input_name = args.seq_input_name
	structure_input_matching_name = args.structure_input_matching_name
	structure_input_path = args.structure_input_path
	structure_input_matching_path = args.structure_input_matching_path
	feature_selected_boolean_path = args.feature_selected_boolean_path
	experiment_name = args.experiment_name + '_'
	output_path = args.output_path
	output_our_model_name = args.output_our_model_name
	output_sota_name = args.output_sota_name
	seq_specific_name = args.seq_specific_name
	feature_selected_boolean = np.load(feature_selected_boolean_path)
	#our_model = torch.load(our_model_path + our_model_name, map_location = 'cpu')
	#$our_model.eval()
	#our_model.cuda()
	sota_model = torch.load(sota_model_path + sota_model_name, map_location = 'cpu')
	sota_model.eval()
	sota_model.cuda()
	structure_all = np.load(structure_input_path)
	inference_seq_x = np.swapaxes(np.load(seq_input_path + seq_input_name), 1, 2)
	structure_input_matching_index_npy = np.load(seq_input_path + structure_input_matching_name)
	inference_structure_x = structure_all[np.asarray(structure_input_matching_index_npy, dtype = int), :]
	loss_function = nn.BCELoss()
	inference_size = inference_seq_x.shape[0]
	sota_pred = np.zeros((inference_seq_x.shape[0], int(np.sum(feature_selected_boolean))))
	inference_seq_x = torch.from_numpy(inference_seq_x).float()
	inference_structure_x = torch.from_numpy(inference_structure_x).float()
	inference_dataloader = DataLoader(TensorDataset(inference_seq_x, inference_structure_x), batch_size = 64, shuffle = False, drop_last=False)
	sample_index = 0
	for inference_batch_i, (inference_batch_seq_x, inference_batch_structure_x) in enumerate(inference_dataloader):
		inference_batch_size = inference_batch_seq_x.shape[0]
		inference_batch_seq_x = inference_batch_seq_x.cuda()
		inference_batch_structure_x = inference_batch_structure_x.cuda()
		sota_pred_firsthalf = sota_model(inference_batch_seq_x).cpu().detach().numpy()[:, feature_selected_boolean]
		inference_batch_seq_x = torch.from_numpy(np.flip(np.flip(inference_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float().cuda()
		sota_pred_secondhalf = sota_model(inference_batch_seq_x).cpu().detach().numpy()[:, feature_selected_boolean]
		sota_pred[sample_index:int(sample_index+inference_batch_size)] = (sota_pred_firsthalf + sota_pred_secondhalf) / 2.0
		sample_index = sample_index + inference_batch_size
	np.save(output_path + experiment_name + output_sota_name + '_' + seq_specific_name +'_prediction.npy', sota_pred)

if __name__=='__main__':
	main()


