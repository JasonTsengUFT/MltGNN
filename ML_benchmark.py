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
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
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
parser.add_argument('--batch_size', type=int, default=1, help='Batchsize')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--time_inter', type=int, default=1, help='Time inter for predict')
parser.add_argument('--file', type=str, default='carts_c', help='file name for text saving')
parser.add_argument('--shuffle', action='store_true', default=False, help='data shuffle')
parser.add_argument('--label_column', type=str, default='carts_c', help='column for train')
parser.add_argument('--feature_num', type=int, default=36, help='Number of features.')  #old_data:36, new_data:64

args = parser.parse_args()
path='datanew/'
def fill_front(a,b):
    if a>0:
        return a
    else:
        return b

cpu = pd.read_csv('{}/cpu.csv'.format(path),dtype=float).fillna(0).iloc[:373,:]
cpu['front_end_c'] = cpu.apply(lambda x: fill_front(x['{pod="front-end-6649c54d45-b5j5h"}'], x['{pod="front-end-6649c54d45-zhlwz"}']),axis=1)
mem = pd.read_csv('{}/memory.csv'.format(path),dtype=float).fillna(0).iloc[:373, :]
mem['front_end_m'] = mem['{pod="front-end-6649c54d45-b5j5h"}'] + mem['{pod="front-end-6649c54d45-zhlwz"}']
cpu_columns = ['time','carts_c','carts-db_c','catalogue_c','catalogue-db_c','front-end-6649c54d45-fl7vx','front-end-6649c54d45-zbm2q','orders_c','orders-db_c',
                   'payment_c','queue-master_c','rabbitmq_c','session-db_c','shipping_c','user_c','user-db_c','front-end_c']
mem_columns = ['time', 'carts_m', 'carts-db_m', 'catalogue_m', 'catalogue-db_m','front-end-6649c54d45-fl7vx','front-end-6649c54d45-zbm2q', 'orders_m', 'orders-db_m',
               'payment_m', 'queue-master_m', 'rabbitmq_m', 'session-db_m', 'shipping_m', 'user_m', 'user-db_m','front-end_m']
cpu.columns = cpu_columns
mem.columns = mem_columns
cpu.drop(columns=['front-end-6649c54d45-fl7vx', 'front-end-6649c54d45-zbm2q'], inplace=True)
mem.drop(columns=['front-end-6649c54d45-fl7vx', 'front-end-6649c54d45-zbm2q'], inplace=True)
dataset = pd.merge(cpu, mem, on='time', how='inner')

diff_columns = dataset.columns.difference(['time'])
# max_min_sacler = MinMaxScaler()
# norm_dataset = pd.DataFrame(max_min_sacler.fit_transform(dataset[diff_columns]), columns=diff_columns, index=dataset.index)
norm_dataset = dataset.drop(columns=['time'])
x = norm_dataset.iloc[:-args.time_inter,:]
df = dataset[diff_columns]
y = df.iloc[args.time_inter:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, shuffle=args.shuffle, random_state=42)

model_xgb = XGBRegressor(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    objective='reg:linear',
    booster='gbtree',
    gamma=0,
    min_child_weight=1,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=1,
    random_state=0
)

model_svr = SVR()

def plot_each_column(model_name,true,pred):
    plt.figure(figsize=(18, 6))  # 设置画布的尺寸
    true = np.array(true)
    pred = np.array(pred)
    acc = 1 - np.mean(abs(pred-true)/true)
    plt.title('XGBoost' + ':' + args.file + '(mean accuracy---{})'.format(round(acc,4)), fontsize=20)  # 标题，并设定字号大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    plt.xlabel(u'time', font1)  # 设置x轴，并设定字号大小
    plt.ylabel(u'Value', font1)  # 设置y轴，并设定字号大小
    plt.tick_params(labelsize=18)
    i = np.arange(true.shape[0])
    plt.plot(i + 1, np.array(pred), linewidth=2, linestyle='-', label='prediction results',
             marker='o')
    plt.plot(i + 1, np.array(true), linewidth=2, linestyle='--', label='True results', marker='+')
    plt.plot(i + 1, abs(np.array(true) - np.array(pred)), linewidth=2,
             linestyle='--', label='error', marker='+')
    plt.legend(prop=font2)  # 图例展示位置，数字代表第几象限
    plt.savefig('{}min/{}/{}.jpg'.format(args.time_inter,file, model_name))
    # plt.show()  # 显示图像
    plt.cla()
    plt.close("all")
    return acc

def run(file):
    re_list = [file]
    y_train_one = np.array(y_train[file])
    y_test_one = np.array(y_test[file])
    model_names = ['xgboost', 'svr']
    for i, model_train in enumerate([model_xgb, model_svr]):
        model_train.fit(x_train, y_train_one)
        ypred = model_train.predict(x_test)
        ypred = pd.DataFrame(ypred)
        target = pd.DataFrame(y_test_one)
        # ypred.to_excel('{}min/{}/Results_{}.xlsx'.format(args.time_inter, file, model_names[i]))
        # target.to_excel('{}min/{}/Target_{}.xlsx'.format(args.time_inter, file, model_names[i]))
        acc = plot_each_column(model_names[i], target, ypred)
        rmse = np.sqrt(MSE(np.array(target,), np.array(ypred)))
        # print('{}----ACC:{}----RMSE:{}'.format(model_names[i], round(acc,4),round(rmse,4)))
        # re_list.append(file)
        re_list.append(model_names[i])
        re_list.append(round(rmse,4))
        re_list.append(round(acc,4))
    return re_list

columns_list = ['carts_c', 'carts-db_c','catalogue_c','catalogue-db_c','front-end_c','orders_c','orders-db_c',
                   'payment_c','queue-master_c','rabbitmq_c','session-db_c','shipping_c','user_c','user-db_c',
                'carts_m', 'carts-db_m', 'catalogue_m', 'catalogue-db_m', 'front-end_m', 'orders_m', 'orders-db_m',
               'payment_m', 'queue-master_m', 'rabbitmq_m', 'session-db_m', 'shipping_m', 'user_m', 'user-db_m']
df_r = pd.DataFrame()
for file in columns_list:
    # print('------------------------------{}--------------------------'.format(file))
    res = run(file)
    res = pd.DataFrame(np.array(res).reshape(1,-1))
    df_r = pd.concat([df_r,res])
df_r.to_excel('{}min/ML_result.xlsx'.format(args.time_inter))



