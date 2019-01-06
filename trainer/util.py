import numpy as np
from enum import Enum

aa_table=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','Z'];

class RunMode(Enum):
    TRAIN=1
    VALIDATE=2
    TEST=3
    UNLABEL=4

def calc_acc1d(pred, y):
    pred_y = np.argmax(pred, axis=-1)
    pos = np.sum(pred_y == y)

    return 1.0 * pos / y.shape[0]

def index_aa(index):
    if index not in range(20):
        return '-'
    return aa_table[index] 

def index_aa_str(index):
    res=[]
    for i in index:
        res.append(index_aa(i))
    return ''.join(res)

