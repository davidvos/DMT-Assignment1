import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset


class MoodDataset(Dataset):
    def __init__(self, X, y):
        X = Variable(torch.Tensor(X.astype(float)))
        y = Variable(torch.Tensor(y.astype(float)))
        y = torch.nn.functional.one_hot(y.to(torch.int64), 10)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]