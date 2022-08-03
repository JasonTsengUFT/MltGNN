import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import argparse
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import xgboost as xgb
from sklearn.svm import SVR

import sys
from torch.autograd import Variable
import scipy.sparse as sp
warnings.filterwarnings("ignore")
NUM = 1
class LstmLayer(nn.Module):
    def __init__(self,input_size,time_step, hidden_size,num_layers,dropout):
        super(LstmLayer, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(hidden_size, 10)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

        self.W1 = nn.Linear(time_step*10, 128)
        nn.init.xavier_uniform_(self.W1.weight, gain=1.414)
        self.relu = nn.SELU()
        self.W2 = nn.Linear(128, time_step)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        self.relu = nn.SELU()

    def forward(self, x):
        r_out, (h_s, h_c) = self.rnn(x.to(torch.float32))
        out = self.hidden_out(r_out)
        out = torch.flatten(out, 1)
        out = out.squeeze()
        out = self.W1(self.relu(out))
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W2(out)
        return out

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        :param h: 输入特征 (node_num,in_features)
        :param adj:  邻接矩阵 (node_num,node_num)
        :return: 输出特征 (node_num ,out_features)
        '''
        h = h.to(torch.float32)
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(torch.from_numpy(adj) > 0, e, zero_vec) 由于adj肯定大于0，所以这里直接用zero_vec代替
        attention = F.softmax(zero_vec, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class EncoderGraph2(nn.Module):
    def __init__(self, dropout, alpha, nheads, train_data_columns, time_steps):
        super(EncoderGraph2, self).__init__()
        self.Front_end_features = ['front-end_c', 'front-end_m','session-db_c','session-db_m']
        self.Order_features = ['orders_c', 'orders-db_c','orders_m','orders-db_m']
        self.User_features = ['user_c', 'user-db_c','user_m','user-db_m']
        self.Catalogue_features = ['catalogue_c','catalogue-db_c','catalogue_m','catalogue-db_m']
        self.Cart_features = ['carts_c', 'carts-db_c','carts_m','carts-db_m']
        self.Payment_features = ['payment_c','payment_m']
        self.Shipping_features = ['shipping_c', 'shipping_m']
        self.Queue_features = ['queue-master_c', 'queue-master_m','rabbitmq_c','rabbitmq_m']

        # '------------------------------------------------newdataset-------------------------------------------------'
        # self.Front_end_features = ['front-end_c', 'front-end_m', 'session-db_c', 'session-db_m', 'front-end_i',
        #                            'front-end_n', 'session-db_i', 'session-db_n']
        # self.Order_features = ['orders_c', 'orders-db_c', 'orders_m', 'orders-db_m', 'orders_i', 'orders-db_i',
        #                        'orders_n', 'orders-db_n']
        # self.User_features = ['user_c', 'user-db_c', 'user_m', 'user-db_m', 'user_i', 'user-db_i', 'user_n',
        #                       'user-db_n']
        # self.Catalogue_features = ['catalogue_c', 'catalogue-db_c', 'catalogue_m', 'catalogue-db_m', 'catalogue_i',
        #                            'catalogue-db_i', 'catalogue_n', 'catalogue-db_n']
        # self.Cart_features = ['carts_c', 'carts-db_c', 'carts_m', 'carts-db_m', 'carts_i', 'carts-db_i', 'carts_n',
        #                       'carts-db_n']
        # self.Payment_features = ['payment_c', 'payment_m', 'payment_i', 'payment_n']
        # self.Shipping_features = ['shipping_c', 'shipping_m', 'shipping_i', 'shipping_n']
        # self.Queue_features = ['queue-master_c', 'queue-master_m', 'rabbitmq_c', 'rabbitmq_m', 'queue-master_i',
        #                        'queue-master_n', 'rabbitmq_i', 'rabbitmq_n']

        self.columns = train_data_columns
        self.columns = list(train_data_columns)
        self.subgraph = [self.Front_end_features , self.Order_features, self.User_features, self.Catalogue_features,
                         self.Cart_features, self.Payment_features, self.Shipping_features, self.Queue_features]
        time_step = 50

        self.EncoderLayer_Front_end = LstmLayer(input_size=len(self.Front_end_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_Order = LstmLayer(input_size=len(self.Order_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_User = LstmLayer(input_size=len(self.User_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_Catalogue = LstmLayer(input_size=len(self.Catalogue_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_Cart = LstmLayer(input_size=len(self.Cart_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_Payment = LstmLayer(input_size=len(self.Payment_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_Shipping = LstmLayer(input_size=len(self.Shipping_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)
        self.EncoderLayer_Queue = LstmLayer(input_size=len(self.Queue_features), time_step=time_step, hidden_size=32, num_layers=2, dropout=0.2)

        self.attentions = [GraphAttentionLayer(time_steps, time_steps, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(time_steps * nheads, time_steps, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):

        def feature_selection(x, features):
            index = [self.columns.index(name) for name in features]
            return x[:, :, index]

        Front_end_out = self.EncoderLayer_Front_end(feature_selection(x, self.Front_end_features))
        Order_out = self.EncoderLayer_Order(feature_selection(x, self.Order_features))
        User_out = self.EncoderLayer_User(feature_selection(x, self.User_features))
        Catalogue_out = self.EncoderLayer_Catalogue(feature_selection(x, self.Catalogue_features))
        Cart_out = self.EncoderLayer_Cart(feature_selection(x, self.Cart_features))
        Payment_out = self.EncoderLayer_Payment(feature_selection(x, self.Payment_features))
        Shipping_out = self.EncoderLayer_Shipping(feature_selection(x, self.Shipping_features))
        Queue_out = self.EncoderLayer_Queue(feature_selection(x, self.Queue_features))


        def feature_adj():
            def normalize_adj(mx):
                """Row-normalize sparse matrix"""
                row, col = np.diag_indices_from(mx)
                mx[row, col] = np.zeros(mx.shape[0])
                rowsum = np.array(mx.sum(1))
                r_inv_sqrt = np.power(rowsum, -0.5).flatten()
                r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
                r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
                return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

            adj = pd.read_excel('adj.xlsx')
            adj = np.array(adj.iloc[:,1:])
            adj = normalize_adj(adj)
            new_features = torch.cat([torch.reshape(Front_end_out, (50, 1)), torch.reshape(Order_out, (50, 1)), torch.reshape(User_out, (50, 1)),
                                      torch.reshape(Catalogue_out, (50, 1)), torch.reshape(Cart_out, (50, 1)), torch.reshape(Payment_out, (50, 1)),
                                      torch.reshape(Shipping_out, (50, 1)), torch.reshape(Queue_out, (50, 1))], dim=1)
            # new_features.columns = ['Front_end_','Order','User','Catalogue','Cart','Payment','Shipping','Queue']
            return adj, new_features


        def gat_process(x, adj):
            x = torch.transpose(x,0,1)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.relu(self.out_att(x, adj))
            return x

        adj, new_features = feature_adj()
        super_data = gat_process(new_features, adj)
        new_dataset = torch.transpose(super_data,0,1)
        raw_data = x
        raw_data = raw_data.reshape(raw_data.shape[1], raw_data.shape[2])
        com_dataset = torch.cat([raw_data, new_dataset], dim=1)
        com_dataset = com_dataset.reshape(1, com_dataset.shape[0], com_dataset.shape[1])
        graph_dataset = new_dataset
        return com_dataset, graph_dataset


class Lstm(nn.Module):
    def __init__(self,input_size,time_step, hidden_size,num_layers,dropout):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(hidden_size, 10)  # 最后一个时序的输出接一个全连接层
        self.fc = nn.Linear(hidden_size, 1)

        self.W1 = nn.Linear(time_step*10, 128)
        nn.init.xavier_uniform_(self.W1.weight, gain=1.414)
        self.relu = nn.SELU()
        self.W2 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        self.W3 = nn.Linear(64, NUM)
        nn.init.xavier_uniform_(self.W3.weight, gain=1.414)
        self.relu = nn.SELU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # print(x)
        # r_out, (h_s, h_c) = self.rnn(x.to(torch.float32)) #.to(torch.float32)
        # out = self.hidden_out(r_out)
        # out = torch.flatten(out, 1)
        # out = out.squeeze()
        # out = self.W1(out)
        # out = F.dropout(out, p=0.2, training=self.training)
        # out = self.W2(self.relu(out))
        # out = F.dropout(out, p=0.2, training=self.training)
        # out = self.W3(out)
        ula, (h_out, _) = self.rnn(x, (h_0, c_0))
        # print(h_out.shape)
        # h_out = h_out.view(-1, self.hidden_size)
        output = ula[:,-1,:]
        out = self.fc(output)
        out = out[0]
        return out

class Ful_con(nn.Module):
    def __init__(self,input_size,time_step):
        super(Ful_con, self).__init__()
        self.W1 = nn.Linear(time_step*input_size, 256)
        nn.init.xavier_uniform_(self.W1.weight, gain=1.414)
        self.relu = nn.SELU()
        self.W2 = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        self.W3 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.W3.weight, gain=1.414)
        self.W4 = nn.Linear(64, NUM)
        nn.init.xavier_uniform_(self.W4.weight, gain=1.414)


    def forward(self, x):
        out = torch.flatten(x.to(torch.float32))
        out = out.squeeze()
        out = self.W1(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W2(self.relu(out))
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W3(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W4(out)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, time_step, num_labels=1):
        """
        dmodel:每一条数据特征的个数
        num_labels:最后输入的维度,回归任务设置为1
        """
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        # 定义Transformer的参数，注意nhead必须要被d_model整除  18/3
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # nn.init.uniform(self.transformer_encoder.weight, a=0, b=1)
        self.W1 = nn.Linear(d_model * time_step, 128)
        # nn.init.uniform(self.W1.weight, a=0, b=1)
        nn.init.xavier_uniform_(self.W1.weight, gain=1.414)
        self.relu = nn.SELU()
        self.W2 = nn.Linear(128, 64)
        # nn.init.uniform(self.W2.weight, a=0, b=1)
        nn.init.xavier_uniform_(self.W2.weight, gain=1.414)
        self.W3 = nn.Linear(64, num_labels)
        # nn.init.uniform(self.W3.weight, a=0, b=1)
        nn.init.xavier_uniform_(self.W3.weight, gain=1.414)

    def forward(self, X):
        # X:[batch_size, n_length, d_model]=[batch_size, 20, 18]
        # 因为Transformer需要输入[ n_length,batch_size, d_model]的维度，因此需要转置一下
        input = X.transpose(0, 1)  # input : [n_length, batch_size, d_model]
        out = self.transformer_encoder(input)  # out:[n_length, batch_size, d_model]
        out = out.transpose(0, 1)
        out = torch.flatten(out, 1)  # 将数据拉平
        out = self.W1(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W2(self.relu(out))  # out : [batch_size, n_class]
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.W3(out)
        # print(out.shape)
        # sys.exit()
        return torch.squeeze(out,0)

class LGT(nn.Module):
    def __init__(self, dropout, alpha, nheads, data_columns,time_step,feature_num,sequence_length):
        #args.feature_num,args.sequence_length
        super(LGT, self).__init__()
        self.encdergrpah = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.lstm = Lstm(input_size=feature_num, time_step=sequence_length, hidden_size=32, num_layers=2, dropout=0.2)
        self.transformer = TransformerLayer(d_model=feature_num, nhead=2, time_step=50)
    def forward(self, x):
        x = self.encdergrpah(x)
        # x = self.lstm(x)
        x = self.transformer(x)
        return x

class MltGNN(nn.Module):
    def __init__(self, dropout, alpha, nheads, data_columns,time_step,feature_num,sequence_length):
        super(MltGNN, self).__init__()
        self.encdergrpah1 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.encdergrpah2 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.lstm = Lstm(input_size=feature_num, time_step=sequence_length, hidden_size=32, num_layers=2, dropout=0.2)
        self.Front_end_features = ['front-end_c', 'front-end_m', 'session-db_c', 'session-db_m']
        self.Order_features = ['orders_c', 'orders-db_c', 'orders_m', 'orders-db_m']
        self.User_features = ['user_c', 'user-db_c', 'user_m', 'user-db_m']
        self.Catalogue_features = ['catalogue_c', 'catalogue-db_c', 'catalogue_m', 'catalogue-db_m']
        self.Cart_features = ['carts_c', 'carts-db_c', 'carts_m', 'carts-db_m']
        self.Payment_features = ['payment_c', 'payment_m']
        self.Shipping_features = ['shipping_c', 'shipping_m']
        self.Queue_features = ['queue-master_c', 'queue-master_m', 'rabbitmq_c', 'rabbitmq_m']
        self.columns = list(data_columns)
        self.subgraph = [self.Front_end_features, self.Order_features, self.User_features, self.Catalogue_features,
                         self.Cart_features, self.Payment_features, self.Shipping_features, self.Queue_features]

    def forward(self, xin):
        def feature_selection(x, features):
            index = [self.columns.index(name) for name in features]
            return x[:, :, index].reshape(50,-1)
        _,x = self.encdergrpah1(xin)
        new_feature = torch.zeros((50,1))
        for i,featr in enumerate(self.subgraph):
            add_n = feature_selection(xin,featr)
            add_n = add_n + x[:, [i]].repeat(1,add_n.shape[1])
            new_feature = torch.cat([new_feature,add_n], dim=1)
        new_feature = torch.reshape(new_feature[:,1:],(1,50,28))
        x,_ = self.encdergrpah2(new_feature)
        x = self.lstm(x)
        return x

class MltGNNL(nn.Module):
    def __init__(self, dropout, alpha, nheads, data_columns,time_step,feature_num,sequence_length):
        super(MltGNNL, self).__init__()
        self.encdergrpah1 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.encdergrpah2 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.lstm = Lstm(input_size=feature_num, time_step=sequence_length, hidden_size=32, num_layers=2, dropout=0.2)
        self.Front_end_features = ['front-end_c', 'front-end_m', 'session-db_c', 'session-db_m']
        self.Order_features = ['orders_c', 'orders-db_c', 'orders_m', 'orders-db_m']
        self.User_features = ['user_c', 'user-db_c', 'user_m', 'user-db_m']
        self.Catalogue_features = ['catalogue_c', 'catalogue-db_c', 'catalogue_m', 'catalogue-db_m']
        self.Cart_features = ['carts_c', 'carts-db_c', 'carts_m', 'carts-db_m']
        self.Payment_features = ['payment_c', 'payment_m']
        self.Shipping_features = ['shipping_c', 'shipping_m']
        self.Queue_features = ['queue-master_c', 'queue-master_m', 'rabbitmq_c', 'rabbitmq_m']
        self.columns = list(data_columns)
        self.subgraph = [self.Front_end_features, self.Order_features, self.User_features, self.Catalogue_features,
                         self.Cart_features, self.Payment_features, self.Shipping_features, self.Queue_features]

    def forward(self, xin):
        def feature_selection(x, features):
            index = [self.columns.index(name) for name in features]
            return x[:, :, index].reshape(50,-1)
        x,_ = self.encdergrpah1(xin)
        x = self.lstm(x)
        return x

class MltGNNF(nn.Module):
    def __init__(self, dropout, alpha, nheads, data_columns,time_step,feature_num,sequence_length):
        super(MltGNNF, self).__init__()
        self.encdergrpah1 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.encdergrpah2 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.lstm = Lstm(input_size=feature_num, time_step=sequence_length, hidden_size=32, num_layers=2, dropout=0.2)
        self.Front_end_features = ['front-end_c', 'front-end_m', 'session-db_c', 'session-db_m']
        self.Order_features = ['orders_c', 'orders-db_c', 'orders_m', 'orders-db_m']
        self.User_features = ['user_c', 'user-db_c', 'user_m', 'user-db_m']
        self.Catalogue_features = ['catalogue_c', 'catalogue-db_c', 'catalogue_m', 'catalogue-db_m']
        self.Cart_features = ['carts_c', 'carts-db_c', 'carts_m', 'carts-db_m']
        self.Payment_features = ['payment_c', 'payment_m']
        self.Shipping_features = ['shipping_c', 'shipping_m']
        self.Queue_features = ['queue-master_c', 'queue-master_m', 'rabbitmq_c', 'rabbitmq_m']
        self.columns = list(data_columns)
        self.subgraph = [self.Front_end_features, self.Order_features, self.User_features, self.Catalogue_features,
                         self.Cart_features, self.Payment_features, self.Shipping_features, self.Queue_features]

    def forward(self, xin):
        def feature_selection(x, features):
            index = [self.columns.index(name) for name in features]
            return x[:, :, index].reshape(50,-1)
        _,x = self.encdergrpah1(xin)
        new_feature = torch.zeros((50,1))
        for i,featr in enumerate(self.subgraph):
            add_n = feature_selection(xin,featr)
            add_n = add_n + x[:, [i]].repeat(1,add_n.shape[1])
            new_feature = torch.cat([new_feature,add_n], dim=1)
        new_feature = torch.reshape(new_feature[:,1:],(1,50,28))
        _, x = self.encdergrpah2(new_feature)
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = self.lstm(x)
        return x

class MltGNNT(nn.Module):
    def __init__(self, dropout, alpha, nheads, data_columns,time_step,feature_num,sequence_length):
        super(MltGNNT, self).__init__()
        self.encdergrpah = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.ful = Ful_con(input_size= feature_num, time_step=sequence_length)
        self.encdergrpah1 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.encdergrpah2 = EncoderGraph2(dropout, alpha, nheads, data_columns, time_steps=time_step)
        self.Front_end_features = ['front-end_c', 'front-end_m', 'session-db_c', 'session-db_m']
        self.Order_features = ['orders_c', 'orders-db_c', 'orders_m', 'orders-db_m']
        self.User_features = ['user_c', 'user-db_c', 'user_m', 'user-db_m']
        self.Catalogue_features = ['catalogue_c', 'catalogue-db_c', 'catalogue_m', 'catalogue-db_m']
        self.Cart_features = ['carts_c', 'carts-db_c', 'carts_m', 'carts-db_m']
        self.Payment_features = ['payment_c', 'payment_m']
        self.Shipping_features = ['shipping_c', 'shipping_m']
        self.Queue_features = ['queue-master_c', 'queue-master_m', 'rabbitmq_c', 'rabbitmq_m']
        self.columns = list(data_columns)
        self.subgraph = [self.Front_end_features, self.Order_features, self.User_features, self.Catalogue_features,
                         self.Cart_features, self.Payment_features, self.Shipping_features, self.Queue_features]
    def forward(self, xin):
        def feature_selection(x, features):
            index = [self.columns.index(name) for name in features]
            return x[:, :, index].reshape(50, -1)

        _, x = self.encdergrpah1(xin)
        new_feature = torch.zeros((50, 1))
        for i, featr in enumerate(self.subgraph):
            add_n = feature_selection(xin, featr)
            add_n = add_n + x[:, [i]].repeat(1, add_n.shape[1])
            new_feature = torch.cat([new_feature, add_n], dim=1)
        new_feature = torch.reshape(new_feature[:, 1:], (1, 50, 28))
        x , _ = self.encdergrpah2(new_feature)
        x = self.ful(x)
        return x






