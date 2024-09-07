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


class GCN(Module):
    def __init__(self, in_size, hid_size,stocks):
        super(GCN, self).__init__()
        self.in_features = in_size
        self.out_features = hid_size
        self.weight = Parameter(torch.FloatTensor(hid_size, hid_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias = Parameter(torch.zeros(hid_size))
        self.lstm = LSTM(in_size, hid_size, stocks)
        self.lamb = nn.Parameter(torch.ones(1, 1))

    def forward(self, x, adj):
        temporal_embedding = self.lstm(x)
        support = torch.matmul(adj, temporal_embedding)
        relational_embedding = torch.matmul(torch.unsqueeze(support, dim=-2),
                                            torch.unsqueeze(self.weight, dim=0)).squeeze() + self.bias
        st_embedding = temporal_embedding + (self.lamb) * relational_embedding
        return st_embedding


class Relation_Attention(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(Relation_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)

    def forward(self, inputdata):
        attention_temp = torch.matmul(inputdata, self.W)
        attention = torch.relu(torch.matmul(attention_temp, torch.transpose(attention_temp, -2, -1)))
        return attention


class Self_attention(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(Self_attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.Q_fc = nn.Linear(out_features, out_features)
        self.K_fc = nn.Linear(out_features, out_features)
        self.V_fc = nn.Linear(out_features, out_features)

    def forward(self, inputdata):
        Q = self.Q_fc(inputdata)
        K = self.K_fc(inputdata)
        V = self.V_fc(inputdata)
        Scores = torch.matmul(Q, torch.transpose(K, -2, -1))
        Scores_softmax = F.softmax(Scores, dim=1)
        Market_Signals = torch.matmul(Scores_softmax, V)
        return Market_Signals


class ADGAT(nn.Module):
    def __init__(self, in_features, out_features, stocks,dropout=.2):
        super(ADGAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.attribute = AttributeGate(in_features, out_features, dropout)
        self.self_attention = Self_attention(in_features, out_features, dropout)
        self.relation_attention_ind = Relation_Attention(out_features, out_features, dropout)
        self.W = nn.Parameter(torch.empty(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.lstm = LSTM(in_features, out_features, stocks)

    def forward(self, inputdata, A=None):
        temporal_embedding = self.lstm(inputdata)
        market_Signals = self.self_attention(temporal_embedding)
        relation = self.relation_attention_ind(market_Signals)
        H = torch.matmul(torch.matmul(relation, market_Signals), self.W)

        return H


class AttributeGate(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(AttributeGate, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(2 * in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.b = nn.Parameter(torch.empty(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data)

    def forward(self, h):
        attribute_gate = self._prepare_attentional_mechanism_input(h)
        attribute_gate = torch.tanh(attribute_gate.add(self.b))
        attribute_gate = F.dropout(attribute_gate, self.dropout, training=self.training)
        return attribute_gate

    def _prepare_attentional_mechanism_input(self, h):
        input_l = torch.matmul(h, self.W[:self.in_features, :])
        input_r = torch.matmul(h, self.W[self.in_features:, :])
        return input_l.unsqueeze(1) + input_r.unsqueeze(2)


if __name__ == '__main__':
    in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    model = AttributeGate(in_feat, out_feat, dropout=.5)
    x = torch.ones((66, stocks, in_feat))
    out = model(x)
    print(out.shape)
