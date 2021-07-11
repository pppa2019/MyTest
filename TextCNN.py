import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset,Dataset
import numpy as np

class textCNN(nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        self.args = args
        self.vocab = args['embedding_num']
        self.dim = args['embed_dim']
        class_num = args['class_num']
        input_channel = 1
        kernel_num = args['kernel_num']
        kernel_size_list = args['kernel_size_list']

        self.embedding = nn.Embedding(self.vocab, self.dim)
        self.convs = nn.ModuleList([nn.Conv1d(1, kernel_num, (K)) for K in kernel_size_list])
        self.dropout = nn.Dropout(args['dropout'])
        self.fc = nn.Linear(len(kernel_size_list)*kernel_num, class_num)
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


class load_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def train(model, opt, loss_function, X_train, y_train):
    """
    @param model: 模型
    @param opt: 优化器
    @param loss_function:损失函数
    @param X_train
    @param y_train
    @return: 训练的平均正确率
    """

    model.train()
    epoch = 10
    train_dataset = load_dataset(X_train, y_train)
    data_loader = DataLoader(dataset=train_dataset,
                             batch_size=128,
                             shuffle=True)
    for i in range(epoch):
        avg_acc = []
        for x_batch, y_batch in data_loader:
            #x_batch = torch.LongTensor(x_batch)
            #y_batch = torch.tensor(y_batch).long().squeeze()
            #print(x_batch.max())
            pred = model(x_batch)
            #print(pred.shape)
            _, pred_int = torch.max(pred, dim=1)
            acc = accuracy_score(y_batch, pred_int)
            # print(acc)
            avg_acc.append(acc)
            # print('acc:',acc)
            # print('y batch:',y_batch)
            # print('prediction:', pred)
            loss = loss_function(pred, y_batch.long())
            opt.zero_grad()
            loss.backward()
            opt.step()                      
        avg_acc = np.array(avg_acc).mean()
        print('epoch', i, avg_acc)
    return avg_acc

def evaluate(model, X_train, y_train):
    """
    @param model: 模型
    @param X_train
    @param y_train
    @return: 模型训练在当前测试集的结果
    """
    avg_acc = []
    prediction = []
    model.eval()
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.tensor(y_train).long().squeeze())
    data_loader = DataLoader(dataset=train_dataset,
                             batch_size=256,
                             shuffle=True,
                             drop_last=True)
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            #x_batch = torch.LongTensor(x_batch)
            #y_batch = torch.tensor(y_batch).long().squeeze()
            pred = model(x_batch)
            _, pred_int = torch.max(pred, dim=1)
            acc = accuracy_score(pred_int, y_batch)
            avg_acc.append(acc)
            prediction.extend(pred_int)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc, prediction