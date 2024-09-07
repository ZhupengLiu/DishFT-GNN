import time

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os

from tqdm import tqdm

from DataProcess import *
from evaluate import evaluate, mean_and_variance
from utils import *
import matplotlib.pyplot as plt

from DishFTGNN.model import Student, Teacher

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=50,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=20,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=32, )

parser.add_argument('--task', type=str, default='future', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--T', type=int, default=20, help='20 10 5')

parser.add_argument('--expect', type=float, default=0.04, help='0.04, 0.025, 0.015')

parser.add_argument('--model', type=str, default='TGC')



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class EarlyStopping:
    def __init__(self, args,device,stocks,patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.best_model = Student(args.in_size, hidden_feat, time_length, stocks, args.model, args.task).to(DEVICE).float()
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.trace_func = print

    def __call__(self, val_loss, model, epoch=0):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
        elif ls > self.best_ls:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0

def test(model, tmodel, dataloader, A):
    predicts = None
    labels = None
    model.eval()
    tmodel.eval()

    for data, label in dataloader:
        out, embedding = smodel(data['indicator'], A, tmodel)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    print("ACC:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
          'MCC:{:.4f}'.format(matthews_corrcoef(predicts, labels)))
    return accuracy_score(predicts, labels),matthews_corrcoef(predicts, labels)


def train(args, smodel, tmodel, train_loader):
    for epoch in range(args.epochs):
        train_losses, val_losses = [], []
        for data, label in tqdm(train_loader):
            smodel.train()
            optimizer.zero_grad()
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            sout, sembedding = smodel(data['indicator'], A, tmodel)
            tout, tembedding = tmodel(data['indicator'], data['future'].transpose(-2, -1), A)
            diss_loss = dishcri(sembedding.reshape(-1,sembedding.shape[-1]), tembedding.reshape(-1,sembedding.shape[-1]))
            _, _, c = sout.shape
            sout = sout.reshape(-1, c)
            label = label.reshape(-1)
            pred_loss = criterion(sout, label)
            loss = pred_loss + args.alpha * diss_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        with torch.no_grad():
            smodel.eval()
            for data, label in val_loader:
                sout, sembedding = smodel(data['indicator'], A, tmodel)
                tout, tembedding = tmodel(data['indicator'], data['future'].transpose(-2, -1), A)
                diss_loss = dishcri(sembedding.reshape(-1, sembedding.shape[-1]),
                                    tembedding.reshape(-1, sembedding.shape[-1]))
                _, _, c = sout.shape
                sout = sout.reshape(-1, c)
                label = label.reshape(-1)
                pred_loss = criterion(sout, label)
                loss = pred_loss + args.alpha * diss_loss
                val_losses.append(loss.item())

        print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}'.format(epoch + 1, np.mean(train_losses),
                                                                        np.mean(val_losses)))

        early_stopping(np.mean(val_losses), smodel, epoch)
        if early_stopping.early_stop:
            print("Early stopping with best_score:{}".format(-early_stopping.best_ls))
            break
        if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
            break


if __name__ == '__main__':
    # set_random_seed(1)
    ACC = []
    MCC = []

    for i in range(4):
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
        A = torch.from_numpy(np.load('sp100_graph_relation.npy')).float().to(DEVICE)
        n1, n2 = A.shape
        A = A - torch.eye(n1).to(A.device)
        train_loader, val_loader, test_loader = LoadData('SP100.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.T, args.expect, args.device)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'future':
            criterion = nn.CrossEntropyLoss()
        if args.task == 'price':
            criterion = nn.MSELoss()
        dishcri = HSIC()
        smodel = Student(args.in_size, hidden_feat, time_length, stocks, args.model, args.task)
        smodel.cuda(device=DEVICE)
        smodel = smodel.float()

        tmodel = Teacher(args.in_size, hidden_feat, stocks, args.model, args.task, args.T)
        tmodel.load_state_dict(torch.load('teacher_model.pth'))
        tmodel.cuda(device=DEVICE)
        tmodel.eval()

        early_stopping = EarlyStopping(args, DEVICE, stocks)

        optimizer = optim.Adam(smodel.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               patience=30,
                                                               factor=0.1,
                                                               verbose=True)

        train(args, smodel, tmodel, train_loader)

        acc,mcc=test(early_stopping.best_model, tmodel, test_loader, A)
        ACC.append(acc)
        MCC.append(mcc)
    print(np.mean(ACC),np.std(ACC))
    print(np.mean(MCC),np.std(MCC))
    #     a, b, c, investment_price = evaluate(early_stopping.best_model, tmodel, test_loader, args.T, args.expect, A)
    #     roi.append(a)
    #     sharp.append(b)
    #     mdd.append(c)
    #     investment_price = np.array(investment_price)
    #     if i == 0:
    #         invest = investment_price
    #     else:
    #         invest += investment_price
    #
    # print(mean_and_variance(roi))
    # print(mean_and_variance(sharp))
    # print(mean_and_variance(mdd))
    # invest = invest / 5
    # print(invest.tolist())
    # plt.plot(invest, linestyle='-', color='b', label='Data')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
