import numpy as np
import argparse
import glob
import util

def calc_acc(path):
    data=np.load(path)
    target=data['target']
    pred=data['pred']
    return util.calc_acc1d(pred, target)

def calc_cv(base_dir, name):
    cv_dir = glob.glob('{}/cv*'.format(base_dir))
    acc=[]
    for dir_ in cv_dir:
        path='{}/{}.npz'.format(dir_, name)
        acc_=calc_acc(path)
        acc.append(acc_)
    return np.array(acc)

def calc_consensus(base_dir, name):
    cv_dir = glob.glob('{}/cv*'.format(base_dir))
    preds=[]
    for dir_ in cv_dir:
        path='{}/{}.npz'.format(dir_, name)
        data=np.load(path)
        target=data['target']
        pred=data['pred']
        preds.append(pred)
    preds=np.array(preds)
    ave_pred=np.mean(preds, axis=0)
    acc=util.calc_acc1d(ave_pred, target)
    with open('{}/avg/{}.cnn'.format(base_dir, name), 'w') as f:
        aa_str=util.index_aa_str(target)
        f.write('tgt {}\n'.format(aa_str))
        aa_str=util.index_aa_str(np.argmax(ave_pred, axis=-1))
        f.write('avg {}\n'.format(aa_str))
        for i in np.arange(preds.shape[0]):
            aa_str=util.index_aa_str(np.argmax(preds[i], axis=-1))
            f.write('cv{} {}\n'.format(i, aa_str))
    return acc

def calc_consensus2(base_dir, name):
    cv_dir = glob.glob('{}/cv*'.format(base_dir))
    preds=[]
    for dir_ in cv_dir:
        path='{}/{}.npz'.format(dir_, name)
        data=np.load(path)
        target=data['target']
        pred=data['pred']
        preds.append(np.argmax(pred, axis=-1))
    preds=np.array(preds)
    ave_pred=[]
    for i in np.arange(preds.shape[1]):
        pred=preds[:,i]
        (values,counts) = np.unique(pred,return_counts=True)
        ind=np.argmax(counts)
        ave_pred.append(values[ind])
    ave_pred=np.array(ave_pred)
    acc=np.sum(target==ave_pred) * 1.0 / target.shape[0]
    return acc

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, type=str)
    parser.add_argument('--input_list', required=True, type=str)
    args=parser.parse_args()
    all_acc=[]
    with open(args.input_list) as f:
        for line in f:
            name=line.strip()
            #acc=calc_cv(args.base_dir, name)
            acc=calc_consensus(args.base_dir, name)
            #acc=calc_consensus2(args.base_dir, name)
            print name, np.mean(acc)
            all_acc.append(acc)
        print 'total= {} ave_acc= {}'.format(len(all_acc), np.mean(np.array(all_acc)))

