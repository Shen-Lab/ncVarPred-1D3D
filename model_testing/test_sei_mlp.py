import numpy as np
from metrics import *
from torch.utils.data import TensorDataset, DataLoader
from sei import *
from sei_mlp import *
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type = str)
	parser.add_argument('--seq_label_path', type = str)
	parser.add_argument('--structure_input_path', type = str)
	parser.add_argument('--structure_input_matching_path', type = str)
	parser.add_argument('--output_path', type = str)
	parser.add_argument('--output_model_name', type = str)
	parser.add_argument('--model_version', type = str)
	args = parser.parse_args()
	return args

def main():
	print('testinging start')
	args = parse_arguments()
	model_path = args.model_path
	seq_label_path = args.seq_label_path
	structure_input_path = args.structure_input_path
	structure_input_matching_path = args.structure_input_matching_path
	output_path = args.output_path
	output_model_name = args.output_model_name
	model_version = args.model_version
	bce_loss_threshold = 1e-12
	model = torch.load(model_path)
	model.eval()
	model.cuda()
	valid_seq_x = torch.from_numpy(np.load(seq_label_path + 'seq_validation.npy')).float()
	valid_structure_index = np.asarray(np.load(structure_input_matching_path + 'validation_matching_index.npy'), dtype = int)
	valid_y = torch.from_numpy(np.load(seq_label_path + 'label_validation.npy'))
	test_section_chr = np.load(structure_input_matching_path + 'testing_section_chr.npy')
	test_section_index = np.load(structure_input_matching_path + 'testing_section_index.npy')
	structure_all = np.load(structure_input_path)
	valid_structure_x = torch.from_numpy(structure_all[valid_structure_index, :]).float()
	print('testing data loading finished')
	section_size = len(test_section_chr)
	section_index = np.arange(section_size)
	valid_pred = np.zeros((8000, 21907))
	test_pred = np.zeros((227355, 1153))
	valid_size = valid_pred.shape[0] // 2
	test_size = test_pred.shape[0]
	valid_label = np.zeros((4000, 21907))
	test_label = np.zeros(test_pred.shape)	
	valid_sample_index = 0
	valid_dataloader = DataLoader(TensorDataset(valid_seq_x, valid_structure_x, valid_y), batch_size = 100, shuffle = False, drop_last=False)
	for valid_batch_i, (valid_batch_seq_x, valid_batch_structure_x, valid_batch_y) in enumerate(valid_dataloader):
		batch_size = valid_batch_seq_x.shape[0]
		valid_batch_seq_x = valid_batch_seq_x.cuda()
		valid_batch_structure_x = valid_batch_structure_x.cuda()
		if(model_version == 'Sei_MLP'):
			valid_pred_firsthalf = model(valid_batch_seq_x, valid_batch_structure_x).cpu().detach().numpy()
		elif(model_version == 'Sei'):
			valid_pred_firsthalf = model(valid_batch_seq_x).cpu().detach().numpy()
		valid_pred[int(valid_sample_index):int(valid_sample_index + batch_size), :] = valid_pred_firsthalf
		valid_batch_seq_x = torch.from_numpy(np.flip(np.flip(valid_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
		valid_batch_seq_x = valid_batch_seq_x.cuda()
		if(model_version == 'Sei_MLP'):
			valid_pred_secondhalf = model(valid_batch_seq_x, valid_batch_structure_x).cpu().detach().numpy()
		elif(model_version == 'Sei'):
			valid_pred_secondhalf = model(valid_batch_seq_x).cpu().detach().numpy()
		valid_pred[int(valid_sample_index + valid_size):int(valid_sample_index + batch_size + valid_size), :] = valid_pred_secondhalf
		valid_label[int(valid_sample_index):int(valid_sample_index + batch_size)] = valid_batch_y
		valid_sample_index = valid_sample_index + batch_size
	valid_loss_bce = bce_loss(valid_pred[0:valid_size], valid_label, bce_loss_threshold) + bce_loss(valid_pred[valid_size:], valid_label, bce_loss_threshold)
	valid_loss_bce = valid_loss_bce / 2
	valid_pred = (valid_pred[0:valid_size] + valid_pred[valid_size:]) / 2
	valid_auroc = []
	valid_auprc = []
	for i in range(21907):
		valid_auroc.append(get_auroc(valid_pred[:, i], valid_label[:, i]))
		valid_auprc.append(get_auprc(valid_pred[:, i], valid_label[:, i]))
	test_auroc = []
	test_auprc = []
	for subset_i in range(19):
		test_sample_index = 0
		for section_i in range(section_size):
			print(section_i)
			test_seq_x = torch.from_numpy(np.load(seq_label_path + test_section_chr[section_index[section_i]] + '_seq_section' + str(int(test_section_index[section_index[section_i]])) + '.npy')).float()
			structure_input_matching_index_npy = np.load(structure_input_matching_path + test_section_chr[section_index[section_i]] + '_section' + str(int(test_section_index[section_index[section_i]])) + '_matching_index.npy')
			test_structure_x = torch.from_numpy(structure_all[np.asarray(structure_input_matching_index_npy, dtype = int), :]).float()
			test_y = torch.from_numpy(np.load(seq_label_path + test_section_chr[section_index[section_i]] + '_label_section' + str(int(test_section_index[section_index[section_i]])) + '.npy')[:, int(subset_i * 1153):int(subset_i * 1153 + 1153)]).float()
			test_dataloader = DataLoader(TensorDataset(test_seq_x, test_structure_x, test_y), batch_size = 128, shuffle = False, drop_last=False)		
			for test_batch_i, (test_batch_seq_x, test_batch_structure_x, test_batch_y) in enumerate(test_dataloader):
				batch_size = test_batch_seq_x.shape[0]
				test_batch_seq_x = test_batch_seq_x.cuda()
				test_batch_structure_x = test_batch_structure_x.cuda()
				if(model_version == 'Sei_MLP'):
					test_pred_firsthalf = model(test_batch_seq_x, test_batch_structure_x).cpu().detach().numpy()[:, int(subset_i * 1153):int(subset_i * 1153 + 1153)]
				elif(model_version == 'Sei'):
					test_pred_firsthalf = model(test_batch_seq_x).cpu().detach().numpy()[:, int(subset_i * 1153):int(subset_i * 1153 + 1153)]
				test_pred[int(test_sample_index):int(test_sample_index + batch_size), :] = test_pred_firsthalf
				test_batch_seq_x = torch.from_numpy(np.flip(np.flip(test_batch_seq_x.cpu().detach().numpy(), 1), 2).copy()).float()
				test_batch_seq_x = test_batch_seq_x.cuda()
				if(model_version == 'Sei_MLP'):
					test_pred_secondhalf = model(test_batch_seq_x, test_batch_structure_x).cpu().detach().numpy()[:, int(subset_i * 1153):int(subset_i * 1153 + 1153)]
				elif(model_version == 'Sei'):
					test_pred_secondhalf = model(test_batch_seq_x).cpu().detach().numpy()[:, int(subset_i * 1153):int(subset_i * 1153 + 1153)]
				test_pred[int(test_sample_index):int(test_sample_index + batch_size), :] = test_pred_secondhalf
				test_label[int(test_sample_index):int(test_sample_index + batch_size)] = test_batch_y
				test_sample_index = test_sample_index + batch_size
		if(test_sample_index != test_size):
			print('test size error')
		test_pred = test_pred / 2
		for i in range(1153):
			test_auroc.append(get_auroc(test_pred[:, i], test_label[:, i]))
			test_auprc.append(get_auprc(test_pred[:, i], test_label[:, i]))
	np.save(output_path + output_model_name + '_bceloss' + str(round(valid_loss_bce, 7)) + '_auroc' + str(round(np.nanmean(np.array(valid_auroc)), 5)) + '_auprc' + str(round(np.nanmean(np.array(valid_auprc)), 5)) + '_auroc' + str(round(np.nanmean(np.array(test_auroc)), 5)) + '_auprc' + str(round(np.nanmean(np.array(test_auprc)), 5)) + '.npy', np.zeros(1))
	np.save(output_path + output_model_name + '_auroc.npy', np.array(test_auroc, dtype = float))
	np.save(output_path + output_model_name + '_auprc.npy', np.array(test_auprc, dtype = float))

if __name__=='__main__':
	main()


