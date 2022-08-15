import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
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
	parser.add_argument('--test_fold', type = str)
	parser.add_argument('--model_name', type = str)
	parser.add_argument('--selected_feature', type = str)
	parser.add_argument('--selected_feature_name', type = str)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	test_fold = args.test_fold
	model_name = args.model_name
	selected_feature = np.load(args.selected_feature)
	selected_feature_name = args.selected_feature_name
	fold_list = list(np.array(np.arange(10), dtype = str))
	fold_list.remove(test_fold)
	snp_subset_list = np.array(['Random_negative_SNP', '31kbp_negative_SNP', '6.3kbp_negative_SNP', '710bp_negative_SNP', '360bp_negative_SNP'])
	for snp_subset_i in snp_subset_list:
		for fold_i in fold_list:
			if((snp_subset_i == snp_subset_list[0]) & (fold_i == fold_list[0])):
				x_diff = np.load('model_prediction/GWAS_' + snp_subset_i + '_' + fold_i + '_' + model_name + '_diff.npy')[:, selected_feature]
				x_fc = np.load('model_prediction/GWAS_' + snp_subset_i + '_' + fold_i + '_' + model_name + '_log_odds_fc.npy')[:, selected_feature]
				x_temp = np.concatenate((x_diff, x_fc), 1)
				x_train = x_temp
				y_train = np.zeros(x_temp.shape[0])
			else:
				x_diff = np.load('model_prediction/GWAS_' + snp_subset_i + '_' + str(fold_list[0]) + '_' + model_name + '_diff.npy')[:, selected_feature]
				x_fc = np.load('model_prediction/GWAS_' + snp_subset_i + '_' + str(fold_list[0]) + '_' + model_name + '_log_odds_fc.npy')[:, selected_feature]
				x_temp = np.concatenate((x_diff, x_fc), 1)
				x_train = np.concatenate((x_train, x_temp))
				y_train = np.concatenate((y_train, np.zeros(x_temp.shape[0])))
	for fold_i in fold_list:
		x_diff = np.load('model_prediction/GWAS_GWAS_Catalog_' + fold_i + '_' + model_name + '_diff.npy')[:, selected_feature]
		x_fc = np.load('model_prediction/GWAS_GWAS_Catalog_' + fold_i + '_' + model_name + '_log_odds_fc.npy')[:, selected_feature]
		x_temp = np.concatenate((x_diff, x_fc), 1)
		x_train = np.concatenate((x_train, x_temp))
		y_train = np.concatenate((y_train, np.zeros(x_temp.shape[0]) + 1))		
	x_train = np.abs(x_train)
	x_mean = np.mean(x_train, 0)
	x_std = np.std(x_train, 0)
	x_train = (x_train - x_mean) / x_std
	sample_weight = (1 - np.mean(y_train)) / np.mean(y_train)
	x_test_diff = np.load('model_prediction/GWAS_' + snp_subset_list[0] + '_' + test_fold + '_' + model_name + '_diff.npy')[:, selected_feature]
	x_test_fc = np.load('model_prediction/GWAS_' + snp_subset_list[0] + '_' + test_fold + '_' + model_name + '_log_odds_fc.npy')[:, selected_feature]
	x_test = np.concatenate((x_test_diff, x_test_fc), 1)
	y_test = np.zeros(x_test.shape[0])
	cluster_name = list(np.repeat(snp_subset_list[0], x_test.shape[0]))
	for snp_subset_i in snp_subset_list[1:]:
		x_test_diff = np.load('model_prediction/GWAS_' + snp_subset_i + '_' + test_fold + '_' + model_name + '_diff.npy')[:, selected_feature]
		x_test_fc = np.load('model_prediction/GWAS_' + snp_subset_i + '_' + test_fold + '_' + model_name + '_log_odds_fc.npy')[:, selected_feature]
		x_test_temp = np.concatenate((x_test_diff, x_test_fc), 1)
		x_test = np.concatenate((x_test, x_test_temp))
		y_test = np.concatenate((y_test, np.zeros(x_test_temp.shape[0])))
		cluster_name = np.concatenate((cluster_name, np.repeat(snp_subset_i, x_test_temp.shape[0])))
	x_test_diff = np.load('model_prediction/GWAS_GWAS_Catalog_' + test_fold + '_' + model_name + '_diff.npy')[:, selected_feature]
	x_test_fc = np.load('model_prediction/GWAS_GWAS_Catalog_' + test_fold + '_' + model_name + '_log_odds_fc.npy')[:, selected_feature]
	x_test_temp = np.concatenate((x_test_diff, x_test_fc), 1)
	x_test = np.concatenate((x_test, x_test_temp))
	y_test = np.concatenate((y_test, np.zeros(x_test_temp.shape[0]) + 1))
	cluster_name = np.concatenate((cluster_name, np.repeat('eQTL', x_test_temp.shape[0])))
	x_test = np.abs(x_test)
	x_test = (x_test - x_mean) / x_std
	model = XGBClassifier(learning_rate = 0.1, scale_pos_weight = sample_weight, reg_alpha = 0, reg_lambda = 10)
	model.fit(x_train, y_train)
	pred = model.predict_proba(x_test)[:, 1]
	np.save('model_prediction/GWAS_' + test_fold + '_' + model_name + '_clf_prediction.npy', pred)
	np.save('model_prediction/GWAS_' + test_fold + '_' + model_name + '_clf_cluster_name.npy', cluster_name)
	auroc_summary = np.zeros(len(snp_subset_list))
	auprc_summary = np.zeros(len(auroc_summary))
	for snp_subset_i in range(len(snp_subset_list)):
		selected_boolean = (cluster_name == snp_subset_list[snp_subset_i]) | (cluster_name == 'eQTL')
		auroc_summary[snp_subset_i] = get_auroc(pred[selected_boolean], y_test[selected_boolean])
		auprc_summary[snp_subset_i] = get_auprc(pred[selected_boolean], y_test[selected_boolean])
	np.save('model_prediction/GWAS_' + test_fold + '_' + model_name + '_clf_' + selected_feature_name + '_auroc.npy', auroc_summary)
	np.save('model_prediction/GWAS_' + test_fold + '_' + model_name + '_clf_' + selected_feature_name + '_auprc.npy', auprc_summary)
	np.save('model_prediction/GWAS_' + test_fold + '_' + model_name + '_clf_' + selected_feature_name + '_trainpr' + str(round(np.mean(y_train), 4)) + '_testpr' + str(round(np.mean(y_test), 4)) + '_auroc' + str(round(np.mean(auroc_summary), 5)) + '_auprc' + str(round(np.mean(auprc_summary), 5)) + '.npy', np.zeros(1))

if __name__=='__main__':
	main()

