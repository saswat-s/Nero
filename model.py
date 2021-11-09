import torch
from torch import nn

class DT(torch.nn.Module):
    def __init__(self, param):
        super(DT, self).__init__()
        self.name = 'DT'
        self.param = param

        self.num_instances = self.param['num_instances']
        self.num_embedding_dim = self.param['num_embedding_dim']

        self.num_outputs = self.param['num_outputs']

        self.embeddings = torch.nn.Embedding(self.num_instances, self.num_embedding_dim)

        self.fc0 = nn.Sequential(nn.BatchNorm1d(self.num_embedding_dim + self.num_embedding_dim),
                                 nn.Linear(in_features=self.num_embedding_dim + self.num_embedding_dim,
                                           out_features=self.num_outputs))

    def forward(self, x):
        # (1) Get embeddings
        x = self.embeddings(x)
        # (2) Permutation Invariant Representations
        x = torch.cat((torch.mean(x, 1), torch.sum(x, 1)), 1)
        return torch.sigmoid(self.fc0(x))
