import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.width',200)
pd.set_option('display.max_columns',200)

def fill_front(df,label):
    df['time'] = df.iloc[:, 0]
    df['carts_{}'.format(label)] = df.iloc[:, 1:4].sum(axis=1)
    df['carts-db_{}'.format(label)] = df.iloc[:,4:6].sum(axis=1)
    df['catalogue_{}'.format(label)] = df.iloc[:, 6:9].sum(axis=1)
    df['catalogue-db_{}'.format(label)] = df.iloc[:, 9]
    df['front-end_{}'.format(label)] = df.iloc[:, 10:25].sum(axis=1)
    df['orders_{}'.format(label)] = df.iloc[:, 25:29].sum(axis=1)
    df['orders-db_{}'.format(label)] = df.iloc[:, 29]
    df['payment_{}'.format(label)] = df.iloc[:, 30]
    df['queue-master_{}'.format(label)] = df.iloc[:, 31:37].sum(axis=1)
    df['rabbitmq_{}'.format(label)] = df.iloc[:, 37]
    df['session-db_{}'.format(label)] = df.iloc[:, 38]
    df['shipping_{}'.format(label)] = df.iloc[:, 39:41].sum(axis=1)
    df['user_{}'.format(label)] = df.iloc[:, 41:45].sum(axis=1)
    df['user-db_{}'.format(label)] = df.iloc[:, 45]
    return df

def load_data(sequence_length, time_inter, path='C:/Pycharm-Experiment/load-prediction/510Q',):
    cpu = pd.read_csv('{}/cpu-data.csv'.format(path), dtype=float).fillna(0)
    mem = pd.read_csv('{}/mem-data.csv'.format(path), dtype=float).fillna(0)

    cpu_f = fill_front(cpu, 'c')
    mem_f = fill_front(mem, 'm')

    cpu_columns = ['time','carts_c', 'carts-db_c', 'catalogue_c','catalogue-db_c', 'orders_c', 'orders-db_c',
                   'payment_c','queue-master_c','rabbitmq_c','session-db_c','shipping_c','user_c','user-db_c','front-end_c']
    mem_columns = ['time', 'carts_m', 'carts-db_m', 'catalogue_m', 'catalogue-db_m', 'orders_m', 'orders-db_m',
               'payment_m', 'queue-master_m', 'rabbitmq_m', 'session-db_m', 'shipping_m', 'user_m', 'user-db_m','front-end_m']

    cpu_f.drop(columns=cpu_f.columns.difference(cpu_columns),inplace=True)
    mem_f.drop(columns=mem_f.columns.difference(mem_columns),inplace=True)

    dataset1 = pd.merge(cpu_f, mem_f, on='time', how='inner')
    # dataset = pd.merge(dataset1,dataset2,on=['time'])
    # dataset.to_excel('{}/dataset_merged.xlsx'.format(path))

    diff_columns = dataset1.columns.difference(['time'])
    max_min_sacler = MinMaxScaler()
    norm_dataset = pd.DataFrame(max_min_sacler.fit_transform(dataset1[diff_columns]), columns=diff_columns, index=dataset1.index)

    seq_array, label_array = data_for_train(norm_dataset, sequence_length, time_inter)
    return seq_array, label_array, max_min_sacler, diff_columns

def data_for_train(df, sequence_length, time_inter):

    def gen_sequence(id_df, seq_length, seq_cols, time_inter):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements - seq_length- time_inter), range(seq_length, num_elements - time_inter)):
            yield data_matrix[start:stop, :].reshape(1,seq_length,data_matrix.shape[1])

    sequence_cols = list(df.columns[:])
    seq_gen = list(gen_sequence(df, sequence_length, sequence_cols, time_inter))
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

    def gen_labels(id_df, seq_length, seq_cols, time_inter):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length+time_inter:num_elements+time_inter, :]

    label_array = gen_labels(df, sequence_length, sequence_cols, time_inter)

    return seq_array, label_array
