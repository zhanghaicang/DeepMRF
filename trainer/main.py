import random
import resnet
import full_resnet
import input_data
import argparse
import tensorflow as tf
from tensorflow.contrib.data import prefetch_to_device
import numpy as np
import logging
import sys
import json
from tensorflow.python.lib.io import file_io
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

def parse_model_config(json_file):
    #the 'open' function can't support goole cloud platform
    with file_io.FileIO(json_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train')
    parser.add_argument('--model_type', type=str,
            choices=['resn1d', 'resn2d', 'resn_mrf', 'full_resn_mrf', 'full_resn1d'],
            default='resn2d')
    parser.add_argument('--op_alg', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--input_config', type=str)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learn_rate', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--l1_reg', type=float, default=0.0)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--job_dir', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--summary_dir', type=str)
    parser.add_argument('--out_model_dir', type=str)
    parser.add_argument('--in_model_path', type=str)
    parser.add_argument('--out_pred_dir', type=str)
    parser.add_argument('--gpu_label', type=int, default=0)
    parser.add_argument('--mrf_1d_reg', type=float, default=-1.0)
    parser.add_argument('--mrf_2d_reg', type=float, default=-1.0)
    parser.add_argument('--cv', type=int, default=-1)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    if args.job_dir is not None and args.mode == 'train':
        os.makedirs(args.job_dir)
        if args.summary_dir is None:
            args.summary_dir = '{}/summary'.format(args.job_dir)
            os.makedirs(args.summary_dir)
        if args.out_model_dir is None:
            args.out_model_dir = '{}/model'.format(args.job_dir)
            os.makedirs(args.out_model_dir)
        if args.log_file is None:
            args.log_file = '{}/run.log'.format(args.job_dir)
        if args.out_pred_dir is None:
            args.out_pred_dir = '{}/pred'.format(args.job_dir)
            os.makedirs(args.out_pred_dir)

    seed=2018
    if int(args.cv) >= 0:
        seed += int(args.cv)

    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if args.mode == 'train':
        log_file_stream=file_io.FileIO(args.log_file,'a')
        fh = logging.StreamHandler(log_file_stream)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logging.info("train_config: {}".format(args))
    with open(args.input_config) as f:
        input_config = json.load(f)
        logging.info("input_config: {}".format(json.dumps(input_config)))

    model_config = parse_model_config(args.model_config)
    logging.info('train_config: {:s}'.format(args))
    logging.info('model_config: {:s}'.format(json.dumps(model_config)))
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    sess_config.allow_soft_placement=True

    with tf.Session(config=sess_config) as sess:
        if args.mode == 'train':
            dataset = input_data.TfRecordDataset(input_config, val_size=0.1, cv_label=args.cv)
            if 'full' in args.model_type:
                resn_ = full_resnet.Resnet(sess, dataset, train_config=args, model_config=model_config)
            else:
                resn_= resnet.Resnet(sess, dataset, train_config=args, model_config=model_config)
            resn_.train()
        elif args.mode == 'test':
            dataset = input_data.TfRecordDataset(input_config)
            if 'full' in args.model_type:
                resn_ = full_resnet.Resnet(sess, dataset, train_config=args, model_config=model_config)
            else:
                resn_ = resnet.Resnet(sess, dataset, train_config=args, model_config=model_config)
            resn_.predict1d(args.output_dir, args.in_model_path)

if __name__ == '__main__':
    main()
