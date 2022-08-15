import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer

class DeepSEA_concatenation(nn.Module):
	def __init__(self, ):
		super(DeepSEA_concatenation, self).__init__()
		self.conv1 = nn.Conv1d(4, 320, kernel_size=8)
		self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
		self.drop1 = nn.Dropout(p = 0.2)
		self.conv2 = nn.Conv1d(320, 480, kernel_size=8)
		self.conv3 = nn.Conv1d(480, 960, kernel_size=8)
		self.drop2 = nn.Dropout(p=0.5)
		self.fc1 = nn.Linear(50880, 925)
		self.fc2 = nn.Linear(925+128, 919)
		self.gat1 = [SpGraphAttentionLayer(768, 250, dropout = 0.2, alpha = 0.2, concat=True) for _ in range(4)]
		#self.gat1 = [GraphAttentionLayer(768, 250, dropout = 0.2, alpha = 0.2, concat=True) for _ in range(4)]
		for i, attention in enumerate(self.gat1):
			self.add_module('gat1_{}'.format(i), attention)
		self.gat2 = [SpGraphAttentionLayer(1000, 100, dropout = 0.2, alpha = 0.2, concat=True) for _ in range(4)]
		#self.gat2 = [GraphAttentionLayer(1000, 100, dropout = 0.2, alpha = 0.2, concat=True) for _ in range(4)]
		for i, attention in enumerate(self.gat2):
			self.add_module('gat2_{}'.format(i), attention)
		self.gat3 = SpGraphAttentionLayer(400, 128, dropout = 0.2, alpha = 0.2, concat = False)
		#self.gat3 = GraphAttentionLayer(400, 128, dropout = 0.2, alpha = 0.2, concat = False)
	def forward(self, seq_input, node_input, adj_input, index_input):
		seq_output = self.conv1(seq_input)
		seq_output = nn.functional.relu(seq_output)
		seq_output = self.maxpool(seq_output)
		seq_output = self.drop1(seq_output)
		seq_output = self.conv2(seq_output)
		seq_output = F.relu(seq_output)
		seq_output = self.maxpool(seq_output)
		seq_output = self.drop1(seq_output)
		seq_output = self.conv3(seq_output)
		seq_output = F.relu(seq_output)
		seq_output = self.drop2(seq_output)
		seq_output = seq_output.view(-1, 50880)
		seq_output = self.fc1(seq_output)
		seq_output = F.relu(seq_output)
		structure_output = F.dropout(node_input, 0.2, training = self.training)
		structure_output = torch.cat([att(structure_output, adj_input) for att in self.gat1], dim = 1)
		structure_output = F.dropout(structure_output, 0.2, training = self.training)
		structure_output = torch.cat([att(structure_output, adj_input) for att in self.gat2], dim = 1)
		structure_output = F.dropout(structure_output, 0.2, training = self.training)
		structure_output = F.elu(self.gat3(structure_output, adj_input))
		structure_output = torch.matmul(index_input, structure_output)
		concatenated_output = torch.cat((seq_output, structure_output), dim = 1)
		concatenated_output = F.relu(concatenated_output)
		concatenated_output = self.fc2(concatenated_output)
		concatenated_output = torch.sigmoid(concatenated_output)
		return concatenated_output

