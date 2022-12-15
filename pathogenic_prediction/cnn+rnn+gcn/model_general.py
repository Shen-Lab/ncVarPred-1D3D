import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSEA(nn.Module):
	def __init__(self, ):
		super(DeepSEA, self).__init__()
		self.conv1 = nn.Conv1d(4, 320, kernel_size=8)
		self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
		self.drop1 = nn.Dropout(p = 0.2)
		self.conv2 = nn.Conv1d(320, 480, kernel_size=8)
		self.conv3 = nn.Conv1d(480, 960, kernel_size=8)
		self.drop2 = nn.Dropout(p=0.5)
		self.fc1 = nn.Linear(50880, 925)
		self.fc2 = nn.Linear(925, 919)
	def forward(self, input):
		output = self.conv1(input)
		output = nn.functional.relu(output)
		output = self.maxpool(output)
		output = self.drop1(output)
		output = self.conv2(output)
		output = F.relu(output)
		output = self.maxpool(output)
		output = self.drop1(output)
		output = self.conv3(output)
		output = F.relu(output)
		output = self.drop2(output)
		output = output.view(-1, 50880)
		output = self.fc1(output)
		output = F.relu(output)
		output = self.fc2(output)
		output = torch.sigmoid(output)
		return output

class DanQ(nn.Module):
	def __init__(self, ):
		super(DanQ, self).__init__()
		self.conv1 = nn.Conv1d(4, 320, kernel_size=26)
		self.maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
		self.drop1 = nn.Dropout(p=0.2)
		self.bilstm = nn.LSTM(320, 320, num_layers=1, batch_first=True, bidirectional=True)
		self.drop2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(48000, 925)
		self.fc2 = nn.Linear(925, 919)
	def forward(self, input):
		output = self.conv1(input)
		output = F.relu(output)
		output = self.maxpool(output)
		output = self.drop1(output)
		output = output.transpose(0, 1).transpose(0, 2)
		output, _ = self.bilstm(output)
		output = output.transpose(0, 1)
		output = output.contiguous().view(output.size(0), 48000)
		output = self.drop2(output)
		output = self.fc1(output)
		output = F.relu(output)
		output = self.fc2(output)
		output = torch.sigmoid(output)
		return output

