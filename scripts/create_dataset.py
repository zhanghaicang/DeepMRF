import sys
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import random
import tensorflow as tf

PADDING_FULL_LEN = 500
PADDING_FULL_LEN = 1000

def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=np.reshape(value,-1)))

def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(value,-1)))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_dataset(name_list, oned_feature_dir,\
        twod_feature_dir, output_prefix, chunk_size=1024, mean_std_path=None):
    names = pd.read_csv(name_list, names=['name'], header=None)
    names = list(names['name'])
    if mean_std_path is not None:
       mean_std=np.load(mean_std_path)
       x1d_mean=mean_std['x1d_mean']
       x1d_std=mean_std['x1d_std']
       x2d_mean=mean_std['x2d_mean']
       x2d_std=mean_std['x2d_std']
    random.shuffle(names)
    if chunk_size > 0:
        SAMPLES_EACH_CHUNK = chunk_size
        chunk_num = (len(names) - 1) / SAMPLES_EACH_CHUNK + 1
    else:
        SAMPLES_EACH_CHUNK = len(names) / (-1 * chunk_size)
        chunk_num = -1 * chunk_size
    total_example_num = 0
    total_pos_num = 0
    total_neg_num = 0

    pbar = tqdm(total=len(names))
    start = 0
    for i in range(chunk_num):
        with tf.python_io.TFRecordWriter(
            '%s_%s.tfrecord' % (output_prefix, i)) as record_writer:
            if chunk_size > 0:
                start = i * SAMPLES_EACH_CHUNK
                end = min(len(names), (i+1) * SAMPLES_EACH_CHUNK)
            else:
                end = start + len(names) / np.abs(chunk_size) + int(i < len(names) % np.abs(chunk_size))
                print 'start and end', start, end

            X1d = []
            X2d = []
            Y = []
            for name in names[start:end]:
                fea1 = pd.read_csv('{}/{}.1dfeat'.format(oned_feature_dir, name),
                        header=None, sep='\s+')
                L = fea1.shape[0]
                if L > PADDING_FULL_LEN:
                    continue

                x1d = fea1.iloc[:, 4:].values
                y1d = fea1.iloc[:,1].values
                x1d_cos = np.cos(x1d[:,-5:])
                x1d_sin = np.sin(x1d[:,-5:])
                if args.with_cle:
                    x1d = np.concatenate([x1d[:,:-5], x1d_cos, x1d_sin], axis=1)
                else:
                    x1d = np.concatenate([x1d[:,:3], x1d[:,20:22], x1d_cos, x1d_sin], axis=1)

                fea2 = pd.read_csv('{}/{}.2dfeat'.format(twod_feature_dir, name),
                        header=None, sep='\s+')
                assert L == int(np.sqrt(fea2.shape[0]))
                data = fea2.iloc[:,[2,6,7,8]].values
                data[:,1:3]=np.log2(data[:,1:3]+1.0)
                data = data.reshape((L, L, -1))
                x2d = data[:,:,1:]
                y2d = data[:,:,0]
                x2d_new = np.zeros((L,L,4))
                x2d_new[:,:,:3] = x2d[:,:,:3]
                for i in range(L):
                    for j in range(i+1,L):
                        x2d_new[i,j,3]=x2d_new[j,i,2]
                        x2d_new[j,i,3]=x2d_new[i,j,2]
                        x2d_new[j,i,2]=x2d_new[i,j,3]
                #y2d[np.tril_indices(y2d.shape[0], -1)] = -1
                if mean_std_path is not None:
                    x1d = (x1d - x1d_mean) / x1d_std
                    x2d_new = (x2d_new - x2d_mean) / x2d_std
                example = tf.train.Example(features = tf.train.Features(
                    feature={
                        'name': _bytes_feature(name),
                        'size': _int64_feature(L),
                        'x1d' : _bytes_feature(x1d.astype(np.float32).tobytes()),
                        'x2d' : _bytes_feature(x2d_new.astype(np.float32).tobytes()),
                        'y1d' : _bytes_feature(y1d.astype(np.int16).tobytes()),
                        'y2d' : _bytes_feature(y2d.astype(np.int16).tobytes())
                        }))
                record_writer.write(example.SerializeToString())
                pbar.update(1)
            start = end
    pbar.close()

def calc_mean_std(name_list, oned_feature_dir, twod_feature_dir):
    names = pd.read_csv(name_list, names=['name'], header=None)
    names = list(names['name'])
    x1d_dim=15#no cle
    x2d_dim=4
    if args.with_cle:
        x1d_dim += 17
    x1d_sum=np.zeros(shape=[x1d_dim])
    x2d_sum=np.zeros(shape=[x2d_dim])
    x1d_total = 0
    x2d_total = 0
    pbar = tqdm(total=len(names))
    for name in names:
        fea1 = pd.read_csv('{}/{}.1dfeat'.format(oned_feature_dir, name),
                header=None, sep='\s+')
        L = fea1.shape[0]
        if L > PADDING_FULL_LEN:
            continue

        x1d = fea1.iloc[:, 4:].values
        y1d = fea1.iloc[:,1].values
        x1d_cos = np.cos(x1d[:,-5:])
        x1d_sin = np.sin(x1d[:,-5:])

        #with CLE
        if args.with_cle:
            x1d = np.concatenate([x1d[:,:-5], x1d_cos, x1d_sin], axis=1)
        else:
            x1d = np.concatenate([x1d[:,:3], x1d[:,20:22], x1d_cos, x1d_sin], axis=1)

        fea2 = pd.read_csv('{}/{}.2dfeat'.format(twod_feature_dir, name),
                header=None, sep='\s+')
        assert L == int(np.sqrt(fea2.shape[0]))
        x2d = fea2.iloc[:,[6,7,8]].values
        x2d = x2d.reshape((L, L, -1))
        x2d_new = np.zeros((L,L,4))
        x2d_new[:,:,:3] = x2d
        for i in range(L):
            for j in range(i+1,L):
                x2d_new[i,j,3]=x2d_new[j,i,2]
                x2d_new[j,i,2]=x2d_new[i,j,3]
                x2d_new[j,i,3]=x2d_new[i,j,2]
        #y2d[np.tril_indices(y2d.shape[0], -1)] = -1
        x1d_sum += np.sum(x1d, axis=0)
        x2d_sum += np.sum(x2d_new, axis=(0, 1))
        x1d_total += L
        x2d_total += L * L
        pbar.update(1)
    pbar.close()
    x1d_mean = x1d_sum / x1d_total
    x2d_mean = x2d_sum / x2d_total

    #calculate std
    x1d_sum=np.zeros(shape=[15])
    x2d_sum=np.zeros(shape=[4])
    pbar = tqdm(total=len(names))
    for name in names:
        fea1 = pd.read_csv('{}/{}.1dfeat'.format(oned_feature_dir, name),
                header=None, sep='\s+')
        L = fea1.shape[0]
        if L > PADDING_FULL_LEN:
            continue
        x1d = fea1.iloc[:, 4:].values
        x1d_cos = np.cos(x1d[:,-5:])
        x1d_sin = np.sin(x1d[:,-5:])
        if args.with_cle:
            x1d = np.concatenate([x1d[:,:-5], x1d_cos, x1d_sin], axis=1)
        else:
            x1d = np.concatenate([x1d[:,:3], x1d[:,20:22], x1d_cos, x1d_sin], axis=1)

        fea2 = pd.read_csv('{}/{}.2dfeat'.format(twod_feature_dir, name),
                header=None, sep='\s+')
        assert L == int(np.sqrt(fea2.shape[0]))
        x2d = fea2.iloc[:,[6,7,8]].values
        x2d = x2d.reshape((L, L, -1))
        x2d_new = np.zeros((L,L,4))
        x2d_new[:,:,:3] = x2d
        for i in range(L):
            for j in range(i+1,L):
                x2d_new[i,j,3]=x2d_new[j,i,2]
                x2d_new[j,i,3]=x2d_new[i,j,2]
                x2d_new[j,i,2]=x2d_new[i,j,3]
        x1d_sum += np.sum(np.square(x1d-x1d_mean), axis=0)
        x2d_sum += np.sum(np.square(x2d_new-x2d_mean), axis=(0,1))
        pbar.update(1)
    pbar.close()
    x1d_std = np.sqrt(x1d_sum  / x1d_total)
    x2d_std = np.sqrt(x2d_sum / x2d_total)
    print x1d_mean
    print x1d_std
    print x2d_mean
    print x2d_std

    np.savez('feature_mean_std.npz',
            x1d_mean=x1d_mean,
            x2d_mean=x2d_mean,
            x1d_std=x1d_std,
            x2d_std=x2d_std)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op',
            choices=['create_dataset', 'calc_mean_std'], required=True)
    parser.add_argument('--name_list', type=str)
    parser.add_argument('--oned_feature_dir', type=str)
    parser.add_argument('--twod_feature_dir', type=str)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--with_cle', type=bool, default=False)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--mean_std', type=str)

    args = parser.parse_args()
    global args

    seed=2018
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if args.op == 'create_dataset':
        create_dataset(args.name_list, args.oned_feature_dir,\
                args.twod_feature_dir, args.output_prefix, args.chunk_size, args.mean_std)
    if args.op == 'calc_mean_std':
        calc_mean_std(args.name_list, args.oned_feature_dir,\
                args.twod_feature_dir)

if __name__ == '__main__':
    main()
