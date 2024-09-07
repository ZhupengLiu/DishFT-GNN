from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from Prediction import Predict
from DishFTGNN.GCN import GCN, ADGAT
from DishFTGNN.TGC import TGC

warnings.filterwarnings("ignore")


class Student(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, gnn, task):
        super(Student, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size
        self.lamb = nn.Parameter(torch.ones(1, 1))

        if gnn == "GCN":
            self.gnn_model = GCN(in_size, hid_size,stocks)
        elif gnn == "ADGAT":
            self.gnn_model = ADGAT(in_size, hid_size, stocks)
        elif gnn == "TGC":
            self.gnn_model = TGC(in_size, hid_size, stocks)
        else:
            raise Exception("gnn mode error!")


    def forward(self, x, adjmatrix,tmodel):
        sp_embedding =self.gnn_model(x,adjmatrix)
        out = tmodel.pred(sp_embedding)
        return out, sp_embedding



class LinearAttention(nn.Module):
    def __init__(self, node_dim, future, in_dim):
        super(LinearAttention, self).__init__()
        self.layerQ = nn.Linear(future, in_dim)
        self.layerK = nn.Linear(node_dim, in_dim)
        self.layerV = nn.Linear(node_dim, in_dim)
        self.initialize()

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, future, tau=0.5):
        Q = self.layerQ(future)
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)
        attention_score = torch.matmul(Q, K.transpose(-2, -1))
        attention_weight = F.softmax(attention_score * tau, dim=1)
        z = torch.matmul(attention_weight, V)
        return z

class Fusion(nn.Module):
    def __init__(self, d_mode):
        super(Fusion, self).__init__()
        self.layerV = nn.Parameter(torch.FloatTensor(d_mode, d_mode,d_mode))

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()

    def forward(self, h,f,tau=.5):
        Q = torch.unsqueeze(f, -1)
        K = torch.unsqueeze(h, -2)
        attention = torch.matmul(Q, K).squeeze()

        left = torch.unsqueeze(torch.unsqueeze(f,-2),-2)
        right = torch.unsqueeze(torch.unsqueeze(h,-2),-1)
        support=torch.matmul(left,torch.unsqueeze(torch.unsqueeze(self.layerV,0),0))
        V=torch.matmul(support,right).squeeze()

        norm = F.softmax(attention * tau, dim=-1)
        out = torch.matmul(norm, torch.unsqueeze(V,-1)).squeeze()
        return out
class Teacher(nn.Module):
    def __init__(self, in_size, hid_size, stocks, gnn, task, T):
        super(Teacher, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.futureencoding = nn.Linear(T, hid_size)
        self.integrate=Fusion(hid_size)
        if gnn == "GCN":
            self.gnn_model = GCN(in_size, hid_size,stocks)
        elif gnn == "ADGAT":
            self.gnn_model = ADGAT(in_size, hid_size, stocks)
        elif gnn == "TGC":
            self.gnn_model = TGC(in_size, hid_size, stocks)
        else:
            raise Exception("gnn mode error!")

        self.pred = Predict(hid_size, task)

    #                                bnt
    def forward(self, x, future, adjmatrix):
        sp_embedding =self.gnn_model(x,adjmatrix)
        fufeature = self.futureencoding(future)

        future_aware_embedding = self.integrate(sp_embedding, fufeature)
        out = self.pred(future_aware_embedding)
        return out, future_aware_embedding


if __name__ == '__main__':
    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = Student(in_feat, out_feat, time_length, stocks, 'GCN', 'trend')
    # x = torch.ones((66, time_length, stocks, in_feat))
    # out = model(x, torch.ones(66, stocks, stocks))
    # print(out[0].shape)

    in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    model = LinearAttention(in_feat, in_feat, in_feat)
    x = torch.ones((66, stocks, in_feat))
    out = model(x, x)
    print(out.shape)
