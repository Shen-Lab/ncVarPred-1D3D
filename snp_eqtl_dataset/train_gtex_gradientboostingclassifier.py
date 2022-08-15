import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import argparse

def get_auroc(preds, obs):
	fpr, tpr, thresholds  = metrics.roc_curve(obs, preds, drop_intermediate=False)
	auroc = metrics.auc(fpr, tpr)
	return auroc

def get_auprc(preds, obs):
	precision, recall, thresholds  = metrics.precision_recall_curve(obs, preds)
	auprc = metrics.auc(recall, precision)
	return auprc

def parse_arguments():
	parser = argparse.ArgumentParser(prog = 'deepsea tunning')
	parser.add_argument('--model_name', type = str)
	parser.add_argument('--cellline_name', type = str)
	parser.add_argument('--l2reg', type = str)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	model_name = args.model_name
	cellline_name = args.cellline_name
	l2reg = args.l2reg
	x_diff = np.load('model_prediction/GTEx_' + model_name + '_diff.npy')
	x_fc = np.load('model_prediction/GTEx_' + model_name + '_log_odds_fc.npy')
	x = np.concatenate((x_diff, x_fc), 1)
	x = np.abs(x)
	log2afc = np.array(np.load('GTEx_processed/GTEx_' + cellline_name + '_eqtl_log2_afc.npy'), dtype = float)
	y = log2afc >= 0
	kf = KFold(n_splits = 10, random_state = 919, shuffle = True)
	cutoff_list = np.arange(0, 30) / 10
	auroc_summary = np.zeros((len(cutoff_list), 10))
	auprc_summary = np.zeros(auroc_summary.shape)
	for cutoff_i in range(len(cutoff_list)):
		selected_index = np.abs(log2afc) > cutoff_list[cutoff_i]
		x_selected = x[selected_index]
		y_selected = y[selected_index]
		replicate_i = 0
		for train_index, test_index in kf.split(x_selected):
			x_train = x_selected[train_index]
			y_train = y_selected[train_index]
			x_test = x_selected[test_index]
			y_test = y_selected[test_index]
			x_train = np.abs(x_train)
			x_mean = np.mean(x_train, 0)
			x_std = np.std(x_train, 0)
			x_train = (x_train - x_mean) / x_std
			sample_weight = (1 - np.mean(y_train)) / np.mean(y_train)
			model = XGBClassifier(learning_rate = 0.1, scale_pos_weight = sample_weight, reg_alpha = 0, reg_lambda = float(l2reg))
			model.fit(x_train, y_train)
			pred = model.predict_proba(x_test)[:, 1]
			auroc_summary[cutoff_i, replicate_i] = get_auroc(pred, y_test)
			auprc_summary[cutoff_i, replicate_i] = get_auprc(pred, y_test)
			replicate_i += 1
	np.save('model_prediction/GTEx_' + model_name + '_' + l2reg + '_clf_auroc.npy', auroc_summary)
	np.save('model_prediction/GTEx_' + model_name + '_' + l2reg + '_clf_auprc.npy', auprc_summary)
	np.save('model_prediction/GTEx_' + model_name + '_' + l2reg + '_clf_auroc' + str(round(np.mean(auroc_summary), 5)) + '_auprc' + str(round(np.mean(auprc_summary), 5)) + '.npy', np.zeros(1))

if __name__=='__main__':
	main()

