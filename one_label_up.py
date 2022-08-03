import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# from DataProcessing import load_data
from processing510 import load_data
import warnings
import argparse
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import time
from layers import Lstm, MltGNN,MltGNNL,MltGNNF,MltGNNT
import torch.nn.functional as F
import xgboost as xgb
from sklearn.svm import SVR

import sys
from torch.autograd import Variable
import scipy.sparse as sp
warnings.filterwarnings("ignore")
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--sequence_length', type=int, default=50, help='sequence length for time series.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=16, help='Batchsize')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--time_inter', type=int, default=1, help='Time inter for predict')
parser.add_argument('--file', type=str, default='carts_c', help='file name for text saving')
parser.add_argument('--shuffle', action='store_true', default=False, help='data shuffle')
parser.add_argument('--label_column', type=str, default='carts_c', help='column for train')
parser.add_argument('--feature_num', type=int, default=36, help='Number of features.')  #old_data:36, new_data:64


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

x, y, scaler, print_columns = load_data(sequence_length=args.sequence_length, time_inter=args.time_inter)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=40, shuffle=args.shuffle, random_state=42)

train_input, test_input, train_target, test_target = train_test_split(x_train, y_train, test_size=40, shuffle=args.shuffle, random_state=42)

# train_target = pd.DataFrame(scaler.inverse_transform(train_target), columns=print_columns)
# test_target = pd.DataFrame(scaler.inverse_transform(test_target), columns=print_columns)
#
# y_test = pd.DataFrame(scaler.inverse_transform(y_test), columns=print_columns)

train_target = pd.DataFrame(train_target, columns=print_columns)
test_target = pd.DataFrame(test_target, columns=print_columns)

y_test = pd.DataFrame(y_test, columns=print_columns)

#----------------------- dataloader for one feature---------------------------------
train_target_one = np.array(train_target[args.label_column])
test_target_one = np.array(test_target[args.label_column])
y_test_one = np.array(y_test[args.label_column])

mydataset = Data.TensorDataset(torch.tensor(np.array(train_input)), torch.tensor(np.array(train_target_one)))
data_loader = DataLoader(dataset=mydataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

seldataset = Data.TensorDataset(torch.tensor(np.array(test_input)), torch.tensor(np.array(test_target_one)))
seldata_loader = DataLoader(dataset=seldataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

testdataset = Data.TensorDataset(torch.tensor(np.array(x_test)), torch.tensor(np.array(y_test_one)))
test_loader = DataLoader(dataset=testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Train model
def train(batch_x, batch_y, model):
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    model.train()
    optimizer.zero_grad()
    criterion = nn.L1Loss()
    num = 0
    for item in batch_x.chunk(args.batch_size, 0):
        output_grad = model(item)
        if num == 0:
            output_batch = output_grad
        else:
            output_batch = torch.cat([output_batch, output_grad])
        num = num + 1
    output_grad_batch = torch.stack(output_batch.chunk(args.batch_size, 0))
    target = torch.tensor(batch_y, dtype=torch.float32)#.squeeze()
    loss_train = criterion(output_grad_batch, target)
    # print(loss_train)
    # sys.exit()
    results = torch.squeeze(torch.cat([output_grad, target],dim=0))
    # print(output_grad_batch.detach().numpy().reshape(1,-1)[0])
    # print(target.detach().numpy())
    # sys.exit()
    error = np.sqrt(MSE(output_grad_batch.detach().numpy().reshape(1,-1)[0], target.detach().numpy()))
    loss_train.backward()
    optimizer.step()
    return loss_train.data.item(), error, results, model

def acc_f(pre, tru):
    pre = pre.squeeze().numpy()
    tru = tru.squeeze().numpy()
    a = abs(pre-tru)/tru
    return 1-a

# Train model
def train_model(model):
    outDF = pd.DataFrame()
    t = time.time()
    t_total = time.time()
    loss_values = []
    error_values = []
    error_epoch = []
    test_epoch = []
    testAcc_epoch = []
    Result_DF = pd.DataFrame()
    for epoch in range(args.epochs):
        t1 = time.time()
        rmse = []
        for step, (batch_x, batch_y) in enumerate(data_loader):
            loss_batch, error_x, results, Tmodel = train(batch_x, batch_y, model)
            loss_values.append(loss_batch)
            error_values.append(error_x)
            rmse.append(error_x)

        error_epoch.append(
            sum(error_values[epoch * len(data_loader):(epoch + 1) * len(data_loader)]) / len(data_loader))

        RMSEValue_epoch = []
        Acc_epoch = []


        for stept, (batch_xt, batch_yt) in enumerate(seldata_loader):
            num = 0
            for item in batch_xt.chunk(args.batch_size, 0):
                outputt = Tmodel(item)
                if num == 0:
                    output_gradt = outputt
                else:
                    output_gradt = torch.cat([output_gradt, outputt])
                num = num + 1
            output_gradt = torch.stack(output_gradt.chunk(args.batch_size, 0))
            outDF = outDF.append(pd.DataFrame(np.array(output_gradt.detach().numpy())))
            targett = torch.tensor(batch_yt, dtype=torch.float32)


            errort = np.sqrt(MSE(output_gradt.detach().numpy(), targett.detach().numpy()))
            acct = acc_f(output_gradt.squeeze().detach(), targett.detach())
            RMSEValue_epoch.append(errort)
            Acc_epoch += list(acct)
        test_RMSEValue = sum(RMSEValue_epoch) / len(RMSEValue_epoch)
        test_Acc = sum(Acc_epoch) / len(Acc_epoch)
        test_epoch.append(test_RMSEValue)
        testAcc_epoch.append(test_Acc)
        if epoch == 0:
            RM = test_RMSEValue

        if test_RMSEValue <= RM:
            RM = test_RMSEValue
            torch.save(Tmodel, '{}/net_{}.pth'.format(args.file,model._get_name()))

        print('**********Finish:{:04d}'.format(epoch + 1),
              '---loss in this epoch: {:.4f}'.format(
                  sum(loss_values[epoch * len(data_loader):(epoch + 1) * len(data_loader)]) / len(data_loader)),
              '---training RMSE:{:.4f}'.format(
                  sum(error_values[epoch * len(data_loader):(epoch + 1) * len(data_loader)]) / len(data_loader)),
              '---testing RMSE:{:.4f}'.format(test_RMSEValue),
              '---testing ACC:{:.4f}'.format(test_Acc),
              '---time: {:.4f}s'.format(time.time() - t1))
        re_list = [epoch + 1, round(sum(loss_values[epoch * len(data_loader):(epoch + 1) * len(data_loader)]) / len(data_loader),4),
                   round(sum(error_values[epoch * len(data_loader):(epoch + 1) * len(data_loader)]) / len(data_loader),4), round(test_RMSEValue,4),
                   round(test_Acc,4), round(time.time() - t1,4)]
        Result_DF = Result_DF.append(pd.DataFrame(np.array(re_list)))

    FinModel = torch.load('{}/net_{}.pth'.format(args.file,model._get_name()))
    RMSEValue_e = []
    Acc_e = []
    for stept, (batch_xte, batch_yte) in enumerate(test_loader):
        num = 0
        for item in batch_xte.chunk(args.batch_size, 0):
            outtest = FinModel(item)
            if num == 0:
                output_gradte = outtest
            else:
                output_gradte = torch.cat([output_gradte, outtest])
            num = num + 1
        output_gradte = torch.stack(output_gradte.chunk(args.batch_size, 0))
        targette = torch.tensor(batch_yte, dtype=torch.float32)
        errorte = np.sqrt(MSE(output_gradte.detach().numpy(), targette.detach().numpy()))
        accte = acc_f(output_gradte.squeeze().detach(), targette.detach())
        RMSEValue_e.append(errorte)
        Acc_e += list(accte)
    test_RMSEe = sum(RMSEValue_e) / len(RMSEValue_e)
    test_Acce = sum(Acc_e) / len(Acc_e)
    print('*************** final test RMSE:{}, Acc:{}'.format(test_RMSEe,test_Acce))
    fi_list = ['Final_RMSE', round(test_RMSEe,4),'Final_Acc',round(test_Acce,4),'-','-']
    Result_DF = Result_DF.append(pd.DataFrame(np.array(fi_list)))
    Result_DF = np.array(Result_DF).reshape(-1,6)
    Result_DF = pd.DataFrame(Result_DF,columns=['epoch','loss','train_RMSE','test_RMSE','test_Acc','time'])
    Result_DF.to_excel('{}/Result_{}.xlsx'.format(args.file,model._get_name()))

# Model
Model_LSTM= Lstm(input_size=28, #old_data = 28, new_date=56
             time_step=args.sequence_length,
             hidden_size=32,
             num_layers=2,
             dropout=0.2
             )

Model_MltGNN = MltGNN(dropout=0.2,
             alpha=0.2,
             nheads=6,
             data_columns=print_columns,
             time_step=args.sequence_length,
             feature_num=args.feature_num,
             sequence_length=args.sequence_length
             )

Model_MltGNNL = MltGNNL(dropout=0.2,
             alpha=0.2,
             nheads=6,
             data_columns=print_columns,
             time_step=args.sequence_length,
             feature_num=args.feature_num,
             sequence_length=args.sequence_length
             )

Model_MltGNNF = MltGNNF(dropout=0.2,
             alpha=0.2,
             nheads=6,
             data_columns=print_columns,
             time_step=args.sequence_length,
             feature_num=8,
             sequence_length=args.sequence_length)

Model_MltGNNT = MltGNNT(dropout=0.2,
             alpha=0.2,
             nheads=6,
             data_columns=print_columns,
             time_step=args.sequence_length,
             feature_num=args.feature_num,
             sequence_length=args.sequence_length)

class nnLayer(nn.Module):
    def __init__(self,input_size,time_step, hidden_size,num_layers,dropout):
        super(nnLayer, self).__init__()

        self.W1 = nn.Linear(time_step*28, 128)
        nn.init.xavier_uniform_(self.W1.weight, gain=1.414)
        self.relu = nn.SELU()
        self.W2 = nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        self.W3 = nn.Linear(256, 64)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        self.W4 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = out.squeeze()
        out = self.W1(self.relu(out))
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W2(self.relu(out))
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W3(self.relu(out))
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W4(out)
        return out

Model_nn =  nnLayer(input_size=28, #old_data = 28, new_date=56
             time_step=args.sequence_length,
             hidden_size=32,
             num_layers=2,
             dropout=0.2
             )

if __name__ == '__main__':
    # model_list = [Model_LSTM,Model_nn]
    model_list = [Model_LSTM, Model_MltGNN, Model_MltGNNL, Model_MltGNNF, Model_MltGNNT]
    for model_item in model_list:
        print('---------------{}:{}-------------------------------'.format(model_item.__class__.__name__, args.label_column))
        train_model(model_item)
