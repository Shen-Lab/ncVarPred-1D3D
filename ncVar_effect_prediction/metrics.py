from sklearn import metrics
import numpy as np

def get_auroc(preds, obs):
	fpr, tpr, thresholds  = metrics.roc_curve(obs, preds, drop_intermediate=False)
	auroc = metrics.auc(fpr, tpr)
	return auroc

def get_auprc(preds, obs):
	precision, recall, thresholds  = metrics.precision_recall_curve(obs, preds)
	auprc = metrics.auc(recall, precision)
	return auprc

def bce_loss(pred, label, threshold):
	pred[pred < threshold] = threshold
	pred[pred > (1 - threshold)] = 1 - threshold
	return -np.mean(np.log(pred) * label + np.log(1 - pred) * (1 - label))


