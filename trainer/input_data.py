import pandas as pd
import random
import numpy as np
import gc 
from util import RunMode
import glob
import logging

class MemDataset:
    pass

class TfRecordDataset:
    def __init__(self, input_config, val_size=1, cv_label=-1):
        data={}
        if 'train' in input_config:
            train_file_prefix=input_config.get('train', None)
            input_train_files = glob.glob('{}*.tfrecord'.format(train_file_prefix))
            random.shuffle(input_train_files)

            chunk_num = len(input_train_files)
            if cv_label < 0:
                if type(val_size) == float:
                    train_chunk_num = chunk_num - int(np.ceil(chunk_num * val_size))
                else:
                    train_chunk_num = chunk_num - val_size
                data['train'] = input_train_files[:train_chunk_num]
                data['val'] = input_train_files[train_chunk_num:]
            else:
                train=[]
                val=[]
                cv_str = '_{}.tfrecord'.format(np.abs(cv_label))
                for i in input_train_files:
                    if cv_str in i:
                        val.append(i)
                    else:
                        train.append(i)
                random.shuffle(train)
                data['train']=train
                data['val']=val
                logging.info('train and val')
                logging.info(data['train'])
                logging.info(data['val'])
        if 'test' in input_config:
            data['test']={}
            for k, v in input_config['test'].iteritems():
                data['test'][k] = glob.glob('{}*.tfrecord'.format(v))
        self.data=data
        print self.data

    def get_chunks(self, data_name):
        if data_name in ['train', 'val']:
            return self.data.get(data_name, None)
        else:
            return self.data['test'].get(data_name, None)
