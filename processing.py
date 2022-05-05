import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.width',200)
pd.set_option('display.max_columns',200)

def fill_front(a,b):
    if a>0:
        return a
    else:
        return b

def load_data(sequence_length, time_inter, path='datanew',):
    cpu = pd.read_csv('{}/cpu.csv'.format(path),dtype=float).fillna(0).iloc[:373,:]
    cpu['front_end_c'] = cpu.apply(lambda x: fill_front(x['{pod="front-end-6649c54d45-b5j5h"}'], x['{pod="front-end-6649c54d45-zhlwz"}']),axis=1)
    mem = pd.read_csv('{}/memory.csv'.format(path),dtype=float).fillna(0).iloc[:373, :]
    mem['front_end_m'] = mem['{pod="front-end-6649c54d45-b5j5h"}'] + mem['{pod="front-end-6649c54d45-zhlwz"}']
    cpu_columns = ['time','carts_c','carts-db_c','catalogue_c','catalogue-db_c','front-end-6649c54d45-fl7vx','front-end-6649c54d45-zbm2q','orders_c','orders-db_c',
                   'payment_c','queue-master_c','rabbitmq_c','session-db_c','shipping_c','user_c','user-db_c','front-end_c']
    mem_columns = ['time', 'carts_m', 'carts-db_m', 'catalogue_m', 'catalogue-db_m','front-end-6649c54d45-fl7vx','front-end-6649c54d45-zbm2q', 'orders_m', 'orders-db_m',
               'payment_m', 'queue-master_m', 'rabbitmq_m', 'session-db_m', 'shipping_m', 'user_m', 'user-db_m','front-end_m']
    net_columns = ['time', 'carts_n', 'carts-db_n', 'catalogue_n', 'catalogue-db_n', 'front-end-6649c54d45-fl7vx','front-end-6649c54d45-zbm2q', 'orders_n', 'orders-db_n',
                   'payment_n', 'queue-master_n', 'rabbitmq_n', 'session-db_n', 'shipping_n', 'user_n', 'user-db_n','front-end_n']
    io_columns = ['time', 'carts_i', 'carts-db_i', 'catalogue_i', 'catalogue-db_i', 'front-end-6649c54d45-fl7vx','front-end-6649c54d45-zbm2q', 'orders_i', 'orders-db_i',
                   'payment_i', 'queue-master_i', 'rabbitmq_i', 'session-db_i', 'shipping_i', 'user_i', 'user-db_i','front-end_i']
    cpu.columns = cpu_columns
    mem.columns = mem_columns
    cpu.drop(columns=['front-end-6649c54d45-fl7vx', 'front-end-6649c54d45-zbm2q'], inplace=True)
    mem.drop(columns=['front-end-6649c54d45-fl7vx', 'front-end-6649c54d45-zbm2q'], inplace=True)
    dataset1 = pd.merge(cpu, mem, on='time', how='inner')
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
