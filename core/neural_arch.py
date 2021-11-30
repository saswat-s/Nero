import torch
from torch import nn
import math
import torch.nn.functional as F


class DeepSet(torch.nn.Module):
    """

     f( pool ( {g(x1), ... g(xn) }) ), f and g continues funcs, pool sum => universal

     g(.) is encoder
     f(pool(.)) is decoder
        # (1) Get Embeddings
        # (2) Apply a pooling operation/ we used sum due to Deepset paper universal approx. theorem
        # (3)
        # f( pool ( {g(x1), ... g(xn) }) ), f and g continues funcs, pool sum => universal
    """

    def __init__(self, param):
        super(DeepSet, self).__init__()
        self.name = 'DeepSet'
        self.param = param

        self.num_instances = self.param['num_instances']
        self.num_embedding_dim = self.param['num_embedding_dim']

        self.num_outputs = self.param['num_outputs']

        self.embeddings = torch.nn.Embedding(self.num_instances, self.num_embedding_dim)
        # 3 permutation invariant representations for positive and negative examples.
        self.fc0 = nn.Sequential(nn.BatchNorm1d(self.num_embedding_dim),
                                 nn.Linear(in_features=self.num_embedding_dim,
                                           out_features=self.num_outputs))
        self.fc1 = nn.Sequential(nn.BatchNorm1d(self.num_embedding_dim),
                                 nn.Linear(in_features=self.num_embedding_dim,
                                           out_features=self.num_outputs))

    def forward(self, xpos, xneg):
        xpos_score = self.fc0(torch.sum(self.embeddings(xpos), 1))
        xneg_score = self.fc1(torch.sum(self.embeddings(xneg), 1))
        return torch.sigmoid(xpos_score - xneg_score)

    def positive_expression_embeddings(self, tensor_idx_individuals: torch.LongTensor):
        return self.fc0(torch.sum(self.embeddings(tensor_idx_individuals), 1))

    def negative_expression_embeddings(self, tensor_idx_individuals: torch.LongTensor):
        return self.fc1(torch.sum(self.embeddings(tensor_idx_individuals), 1))


class ST(torch.nn.Module):
    """
     f( pool ( {g(x1), ... g(xn) }) ), f and g continues funcs, pool sum => universal

     g(.) is encoder
     f(pool(.)) is decoder
        # (1) Get Embeddings
        # (2) Apply a pooling operation/ we used sum due to Deepset paper universal approx. theorem
        # (3)
        # f( pool ( {g(x1), ... g(xn) }) ), f and g continues funcs, pool sum => universal

    """

    def __init__(self, param):
        super(ST, self).__init__()
        self.name = 'ST'
        self.param = param
        self.num_instances = self.param['num_instances']
        self.num_embedding_dim = self.param['num_embedding_dim']

        self.num_outputs = self.param['num_outputs']

        self.embeddings = torch.nn.Embedding(self.num_instances, self.num_embedding_dim)
        # Like a set wise flattening
        self.set_transformer_negative = SetTransformer(dim_input=self.num_embedding_dim,
                                                       num_outputs=self.num_outputs,
                                                       dim_output=1,
                                                       num_inds=4, dim_hidden=4,
                                                       num_heads=4, ln=False)
        self.set_transformer_positive = SetTransformer(dim_input=self.num_embedding_dim,
                                                       num_outputs=self.num_outputs,
                                                       dim_output=1,
                                                       num_inds=4, dim_hidden=4, num_heads=4, ln=False)

    def forward(self, xpos, xneg):
        # assert xpos.shape == xneg.shape
        # (1) Get embeddings
        # {g(x1)}
        xpos_score = torch.squeeze(self.set_transformer_positive(self.embeddings(xpos)), dim=2)
        xneg_score = torch.squeeze(self.set_transformer_negative(self.embeddings(xneg)), dim=2)
        return torch.sigmoid(xpos_score - xneg_score)

    def get_pos_embeddings(self, xpos):
        return torch.squeeze(self.set_transformer_positive(self.embeddings(xpos)), dim=2)

    def get_neg_embeddings(self):
        return torch.squeeze(self.set_transformer_negative(self.embeddings(xneg)), dim=2)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
