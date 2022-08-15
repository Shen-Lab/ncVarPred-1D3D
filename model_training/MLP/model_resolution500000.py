import torch
import torch.nn as nn
import torch.nn.functional as F

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
		self.structure_fc1 = nn.Linear(6207, 1000)
		self.structure_fc2 = nn.Linear(1000, 400)
		self.structure_fc3 = nn.Linear(400, 128)
	def forward(self, seq_input, adj_input):
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
		structure_output = self.structure_fc1(adj_input)
		structure_output = F.relu(structure_output)
		structure_output = self.structure_fc2(structure_output)
		structure_output = F.relu(structure_output)
		structure_output = self.structure_fc3(structure_output)
		concatenated_output = torch.cat((seq_output, structure_output), dim = 1)
		concatenated_output = F.relu(concatenated_output)
		concatenated_output = self.fc2(concatenated_output)
		concatenated_output = torch.sigmoid(concatenated_output)
		return concatenated_output

