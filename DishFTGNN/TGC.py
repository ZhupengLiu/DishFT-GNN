from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings


warnings.filterwarnings("ignore")

class LSTMgate(nn.Module):
    def __init__(self, input_size, output_size, activation, stocks, ):
        super(LSTMgate, self).__init__()
        self.activation = activation
        self.W = Parameter(torch.FloatTensor(stocks, input_size, output_size))
        self.bias = Parameter(torch.zeros(stocks, output_size))
        self.reset_param(self.W)

    def reset_param(self, x):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.unsqueeze(x, -2)
        return self.activation(torch.matmul(x, self.W).squeeze() + self.bias)


class LSTMcell(nn.Module):
    def __init__(self, in_size, out_size, stocks):
        super(LSTMcell, self).__init__()
        self.in_size = in_size
        self.out_feat = out_size
        self.input = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.output = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.forget = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.candidate = LSTMgate(in_size + out_size, out_size, nn.Tanh(), stocks)

    def forward(self, xt, hidden, ct_1):  # hidden:t-1
        _, N, D = hidden.shape

        it = self.input(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ot = self.output(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ft = self.forget(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))

        chat = self.candidate(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ct = ft * ct_1.expand(len(xt), N, D) + it * chat
        ht = ot * torch.tanh(ct)
        return ht, ct


class LSTM(nn.Module):
    def __init__(self, in_feat, out_feat, stocks):
        super(LSTM, self).__init__()
        self.in_feat = in_feat
        self.hid_size = out_feat
        self.stocks = stocks
        self.lstmcell = LSTMcell(in_feat, out_feat, stocks)

    #              B*T*N*D
    def forward(self, x, hidden=None, c=None):
        h = []
        if hidden == None:
            hidden = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
            c = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)

        for t in range(len(x[0])):
            hidden, c = self.lstmcell(x[:, t], hidden, c)
            h.append(hidden)
        att_ht = hidden
        return att_ht.squeeze()
class TGC(nn.Module):
    def __init__(self, in_size, hid_size, stocks):
        super(TGC, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size
        self.drop=nn.Dropout(.5)
        self.temporal = LSTM(in_size, hid_size, stocks)
        self.relational=GraphModule(hid_size,hid_size)

    def forward(self, x, adjmatrix):
        temporal_embedding = self.temporal(x)
        temporal_embedding=self.drop(temporal_embedding)
        relational_embedding = self.relational(temporal_embedding, adjmatrix)
        relational_embedding=self.drop(relational_embedding)

        return relational_embedding


class GraphModule(nn.Module):
    def __init__(self, infeat, outfeat):
        super().__init__()

        self.g = nn.Linear(infeat + infeat + 1, 1)
        self.weight = Parameter(torch.FloatTensor(infeat, outfeat))
        self.bias = Parameter(torch.FloatTensor(outfeat))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #                 BND
    def forward(self, inputs, relation):
        b, n, d = inputs.shape[0], inputs.shape[1], inputs.shape[2]

        x_expanded = inputs.unsqueeze(1).expand(b, n, n, d)
        x_expanded_transposed = x_expanded.transpose(1, 2)

        relation=relation.unsqueeze(-1).unsqueeze(0).expand(b,n,n,1)

        out = torch.cat((x_expanded, x_expanded_transposed,
                         relation), dim=3)
        res = self.g(out).squeeze()
        output = torch.matmul(torch.matmul(res, inputs), self.weight) + self.bias

        return output



if __name__ == '__main__':
    pass
    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = Student(in_feat, out_feat, time_length, stocks, 'GCN', 'trend')
    # x = torch.ones((66, time_length, stocks, in_feat))
    # out = model(x, torch.ones(66, stocks, stocks))
    # print(out[0].shape)

    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = LinearAttention(in_feat, in_feat, in_feat)
    # x = torch.ones((66, stocks, in_feat))
    # out = model(x, x)
    # print(out.shape)
