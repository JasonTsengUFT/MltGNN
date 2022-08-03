import os
file_list = ['5min']
minute_list = [5]
columns_list = [#'carts_c', 'carts-db_c','catalogue_c','catalogue-db_c','front-end_c','orders_c','orders-db_c',
                #'payment_c','queue-master_c','rabbitmq_c','session-db_c','shipping_c','user_c','user-db_c',
                'carts_m', 'carts-db_m', 'catalogue_m', 'catalogue-db_m', 'front-end_m', 'orders_m',
                'orders-db_m','payment_m', 'queue-master_m', 'rabbitmq_m', 'session-db_m', 'shipping_m', 'user_m', 'user-db_m'
]

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("---  There is this folder!  ---")

for i,fi in enumerate(file_list):
    file_first = '{}'.format(fi)
    minute_time = minute_list[i]
    mkdir(file_first)
    for item in columns_list:
        file = '{}'.format(item)
        mkdir(file_first + '/' + file)
        #--------------deep learning---------------------
        os.system('python ./one_label_up.py --file={} --label_column={} --time_inter={}'.format(file_first + '/' + item, item, minute_time))
        # os.system("python ./plt.py --file={}".format(file_first + '/' + item))
        # os.system("python ./eachnode.py --file={}".format(file_first + '/' + item))

        # --------------machine learning---------------------

        # os.system('python ./ML_benchmark.py --file={} --label_column={} --time_inter={}'.format(file_first + '/' + item, item,minute_time))
