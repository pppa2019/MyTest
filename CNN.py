from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset,Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.vocab = args['embedding_num']
        self.dim = args['embed_dim']
        class_num = args['class_num']
        input_channel = 1
        kernel_num = args['kernel_num']
        kernel_size_list = args['kernel_size_list']
        self.conv1 = nn.Conv1d()

        self.out = nn.Linear(len(kernel_size_list)*kernel_num, class_num)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.float()
        self.embedding = torch.nn.Embedding(x.size(0), x.size(1))
        self.embedding.weight = torch.nn.Parameter(x)
        self.embedding.weight.requires_grad = False
        # x = self.embedding(x)
        x = x.unsqueeze(1)
        x = x.float()
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(x_sub.squeeze(-1), x_sub.shape[-1]) for x_sub in x]
        x = torch.cat(x, 1)
        output = self.fc(x.squeeze(-1))
        return output