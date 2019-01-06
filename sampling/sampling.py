import random
import copy
import numpy as np
import argparse
np.set_printoptions(threshold=np.nan)

def calc_acc(pred, target):
    return 1.0 * np.sum(pred==target) / pred.shape[0]

def evaluate(pred, target):
    acc=calc_acc(pred, target)
    #print 'acc= {:.4f} len= {}'.format(acc, pred.shape[0])
    return acc

def calc_opt_1d_diagonal(prob):
    L = prob.shape[0]
    pred=np.zeros(L)
    for i in range(L):
        z = prob[i,i].reshape((20,20)).diagonal()
        pred[i] = np.argmax(z)
    return pred

def calc_opt_1d_avg(prob):
    z=prob.reshape((prob.shape[0], prob.shape[0], 20, 20))
    z=np.sum(z, axis=3)
    z=np.transpose(z, (0,2,1))
    z=np.sum(z, axis=2)
    for i in range(z.shape[0]):
        zz = prob[i,i].reshape((20,20)).diagonal()
        z[i] -= zz
    pred=np.argmax(z, axis=1)
    return pred

def calc_opt_1d_avg2(prob):
    z=prob.reshape((prob.shape[0], prob.shape[0], 20, 20))
    z=np.sum(z, axis=3)
    z=np.transpose(z, (0,2,1))
    z=np.sum(z, axis=2)
    pred=np.argmax(z, axis=1)
    return pred

def calc_opt_mrf_v1(mrf_1d, mrf_2d):
    pred=np.argmax(mrf_1d, axis=1)
    return pred

def eval_obj(mrf_1d, mrf_2d, target, j, a):
    res = 0.0
    if a < 19:
        res = mrf_1d[j][a]
    for i in range(target.shape[0]):
        if i == j:
            continue
        if a < 19 and target[i] < 19:
            res += mrf_2d[j,i,a,target[i]]
    return res

def eval_obj2(mrf_1d, mrf_2d, target):
    res = 0
    for i in range(target.shape[0]):
        if target[i] < 19:
            res += mrf_1d[i,target[i]]
    for i in range(target.shape[0]):
        for j in range(i+1, target.shape[0]):
            if target[i] < 19 and target[j] < 19:
                res += mrf_2d[i,j,target[i],target[j]]
    return res

def calc_opt_mrf_v2(mrf_1d, mrf_2d, target):
    np.set_printoptions(precision=4, suppress=True)
    print np.mean(np.square(mrf_1d)), np.mean(np.square(mrf_2d))
    print mrf_1d.shape, mrf_2d.shape
    print 'max', np.max(np.max(mrf_2d, axis=-1), axis=-1)
    print '1d'
    print mrf_1d
    print '2d 0, 1'
    print mrf_2d[0,1]
    print '2d 1, 0'
    print mrf_2d[1,0]
    import sys
    sys.exit()

    max_iter=100
    init=calc_opt_mrf_v1(mrf_1d, mrf_2d)
    #prev=copy.copy(init)
    idx=range(target.shape[0])
    cur=copy.copy(init)
    for i in range(max_iter):
        acc=calc_acc(cur, target)
        obj2=eval_obj2(mrf_1d, mrf_2d, cur)
        print 'iter= {} acc= {} obj= {}'.format(i, acc, obj2)
        random.shuffle(idx)
        diff=0
        for j in idx:
            max_a=-1
            max_v=-100000.0
            for a in range(20):
                obj=eval_obj(mrf_1d, mrf_2d, cur, j, a)
                if obj > max_v:
                    max_a = a
                    max_v = obj
            if max_a != cur[j]:
                diff+=1
                cur[j]=max_a
        print 'diff', diff
        if diff == 0:
            break
    #import sys
    #sys.exit()
    return cur

def calc_opt_mrf_v3(mrf_1d, mrf_2d, target):
    def one_try(init):
        obj2 = eval_obj2(mrf_1d, mrf_2d, init)
        acc=calc_acc(init, target)
        #print 'iter= {} max_v= {} acc= {} diff= {}'.format('init', obj2, acc, 0)

        max_iter=100
        idx=range(target.shape[0])
        cur=copy.copy(init)
        for i in range(max_iter):
            random.shuffle(idx)
            diff=0
            for j in idx:
                max_a=-1
                max_v=-100000.0
                for a in range(20):
                    obj=eval_obj(mrf_1d, mrf_2d, cur, j, a)
                    if obj > max_v:
                        max_a = a
                        max_v = obj
                if max_a != cur[j]:
                    diff+=1
                    cur[j]=max_a
            obj2 = eval_obj2(mrf_1d, mrf_2d, cur)
            acc=calc_acc(cur, target)
            #print 'iter= {} max_v= {} acc= {} diff= {}'.format(i, obj2, acc, diff)
            if diff == 0:
                break
        return cur, obj2 

    opt_str = calc_opt_mrf_v1(mrf_1d, mrf_2d)
    opt_v = eval_obj2(mrf_1d, mrf_2d, opt_str)
    acc=calc_acc(opt_str, target)
    print 'only 1d len= {} obj= {} acc= {}'.format(target.shape[0], opt_v, acc)

    opt_str, opt_v = one_try(opt_str)
    acc=calc_acc(opt_str, target)
    print 'round= init obj= {} acc= {}'.format(opt_v, acc)

    rounds=10
    for i in range(0):
        init_str = np.random.randint(20, size=target.shape[0])
        #print init_str
        cur, val = one_try(init_str)
        acc=calc_acc(cur, target)
        print 'round= {} obj= {} acc= {}'.format(i, val, acc)
        if val > opt_v:
            opt_v = val
            opt_str = cur
    print 'max obj= {} acc= {}'.format(opt_v, calc_acc(opt_str, target))
    return opt_str
    
def calc_acc(pred, target):
    return 1.0 * np.sum(pred==target) / pred.shape[0]
    pred=np.argmax(mrf_1d, axis=1)
    return pred

def calc_seq(input_pred):
    data=np.load(input_pred)
    prob=data['pred']
    target=data['y']
    pred=calc_opt_1d_diagonal(prob)
    acc1=evaluate(pred, target)
    pred=calc_opt_1d_avg(prob)
    acc2=evaluate(pred, target)
    pred=calc_opt_1d_avg2(prob)
    acc3=evaluate(pred, target)
    return acc1, acc2, acc3

def calc_mrf_seq(input_pred):
    data=np.load(input_pred)
    mrf_1d=data['mrf_1d']
    mrf_2d=data['mrf_2d']
    mrf_2d_sym=np.transpose(mrf_2d, [1,0, 3, 2])
    mrf_2d=mrf_2d+mrf_2d_sym
    #mrf_2d=0.5*(mrf_2d+mrf_2d_sym)
    target=data['y']

    target_obj=eval_obj2(mrf_1d, mrf_2d, target)
    print 'target obj', target_obj
    pred=calc_opt_mrf_v1(mrf_1d, mrf_2d)
    acc1=evaluate(pred, target)
    pred=calc_opt_mrf_v3(mrf_1d, mrf_2d, target)
    acc2=evaluate(pred, target)
    return acc1, acc2
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_pred', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--input_list', type=str)
    args=parser.parse_args()
    print args
    random.seed(2018)
    random.seed(4)
    if args.input_pred is not None:
        calc_seq(args.input_pred)
    if args.input_dir is not None and args.input_list is not None:
        with open(args.input_list) as f:
            res=[]
            for line in f:
                #if line.strip() !='5ofjA':
                #    continue
                path='{}/{}.pred.npz'.format(args.input_dir, line.strip())
                print line.strip()
                #acc1, acc2, acc3=calc_seq(path)
                #res.append([acc1, acc2, acc3])
                acc1, acc2 = calc_mrf_seq(path)
                res.append([acc1, acc2])
            res=np.array(res)
            print 'avg', np.mean(res, axis=0)
