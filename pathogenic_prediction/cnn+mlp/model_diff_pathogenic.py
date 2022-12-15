import torch
import torch.nn as nn
import torch.nn.functional as F

class finetune_pathogenic(nn.Module):
	def __init__(self, ):
		super(finetune_pathogenic, self).__init__()
		self.finetune_fc1 = nn.Linear(919, 1)
		self.batchnorm1 = nn.BatchNorm1d(919)
	def forward(self, wt_embedding_output, mt_embedding_output):
		diff_output = wt_embedding_output - mt_embedding_output
		output = torch.square(diff_output)
		output = self.batchnorm1(output)
		output = self.finetune_fc1(output)
		output = torch.sigmoid(output)
		return output

