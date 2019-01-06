import random
import layers
import os
import tensorflow as tf
import numpy as np
import tensorflow as tf
import util
from util import RunMode
import logging
np.set_printoptions(threshold=np.nan)

PADDING_FULL_LEN = 500
#PADDING_FULL_LEN = None
#PADDING_FULL_LEN = 250

class Resnet:
    def __init__(self, sess, dataset, train_config, model_config):
        self.sess = sess
        self.dataset = dataset
        self.train_config = train_config
        self.model_config = model_config

        self.input_tfrecord_files = tf.placeholder(tf.string, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.x1d_channel_dim = model_config['1d']['channel_dim']
        self.x2d_channel_dim = model_config['2d']['channel_dim']

    def resn1d(self, x1d, reuse=False):
        with tf.variable_scope('resn_1d', reuse=reuse) as scope:
            act = tf.nn.relu
            filters_1d = self.model_config['1d']['filters']
            kernel_size_1d = self.model_config['1d']['kernel_size']
            block_num_1d = self.model_config['1d']['block_num']

            kernel_initializer = tf.glorot_normal_initializer()
            bias_initializer = tf.zeros_initializer()
            kernel_regularizer = tf.contrib.layers.l1_l2_regularizer(
                    scale_l1=self.train_config.l1_reg, scale_l2=self.train_config.l2_reg)
            bias_regularizer = tf.contrib.layers.l1_l2_regularizer(
                    scale_l1=self.train_config.l1_reg, scale_l2=self.train_config.l2_reg)

            block=layers.resn1d_regular_block_v3
            for i in np.arange(block_num_1d):
                inputs=x1d if i==0 else conv_1d
                conv_1d = block(inputs, act, filters_1d, kernel_size_1d,
                        kernel_initializer, bias_initializer,
                        kernel_regularizer, bias_regularizer,
                        self.training, names='conv_layer_{}'.format(i))

            conv_1d = layers.batch_norm(conv_1d, training=self.training)
            conv_1d = act(conv_1d)
            logits = tf.layers.conv1d(inputs=conv_1d, filters=self.model_config['1d_label_size'],
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
            '''
            lr_logits = tf.layers.conv1d(inputs=inputs, filters=self.model_config['1d_label_size'],
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
        return logits + lr_logits
            '''
        return logits

    def resn_mrf(self, x1d, x2d, y1d, reuse=False):
        with tf.variable_scope('resn_1d_2d', reuse=reuse) as scope:
            act = tf.nn.relu
            filters_1d = self.model_config['1d']['filters']
            kernel_size_1d = self.model_config['1d']['kernel_size']
            block_num_1d = self.model_config['1d']['block_num']

            filters_2d = self.model_config['2d']['filters']
            kernel_size_2d = self.model_config['2d']['kernel_size']
            block_num_2d = self.model_config['2d']['block_num']

            kernel_initializer = tf.glorot_normal_initializer()
            bias_initializer = tf.zeros_initializer()
            kernel_regularizer = tf.contrib.layers.l1_l2_regularizer(
                    scale_l1=0.0, scale_l2=self.train_config.l2_reg)
            bias_regularizer = tf.contrib.layers.l1_l2_regularizer(
                    scale_l1=0.0, scale_l2=self.train_config.l2_reg)

            block_1d=layers.resn1d_regular_block_v3
            for i in np.arange(block_num_1d):
                inputs=x1d if i==0 else conv_1d
                conv_1d = block_1d(inputs, act, filters_1d, kernel_size_1d,
                        kernel_initializer, bias_initializer,
                        kernel_regularizer, bias_regularizer,
                        self.training, names='conv1d_layer_{}'.format(i))
            conv_1d = layers.batch_norm(conv_1d, training=self.training)
            conv_1d = act(conv_1d)
            logits_1d = tf.layers.conv1d(inputs=conv_1d, filters=19,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

            #2d
            #used_1d=tf.concat([x1d[:,:,:3], x1d[:,:,-12:]], axis=2)
            #out_1d = tf.expand_dims(used_1d, axis=3)
            out_1d = tf.expand_dims(x1d, axis=3)
            #out_1d = tf.expand_dims(logits_1d, axis=3)
            ones = tf.ones((1, tf.shape(x1d)[1]))
            left_1d = tf.einsum('abcd,de->abce', out_1d, ones)
            left_1d = tf.transpose(left_1d, perm=[0,1,3,2])
            right_1d = tf.transpose(left_1d, perm=[0,2,1,3])
            #input_2d = tf.concat([x2d, left_1d, right_1d], axis=3)
            #asymmetric
            input_2d = tf.concat([x2d, right_1d], axis=3)

            block_2d=layers.resn2d_regular_block
            for i in np.arange(block_num_2d):
                inputs=input_2d if i==0 else conv_2d
                conv_2d = block_2d(inputs, act, filters_2d, kernel_size_2d,
                        kernel_initializer, bias_initializer,
                        kernel_regularizer, bias_regularizer,
                        self.training, names='conv2d_layer_{}'.format(i))
            conv_2d = layers.batch_norm(conv_2d, training=self.training)
            conv_2d = act(conv_2d)
            logits_2d = tf.layers.conv2d(inputs=conv_2d, filters=19*19,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
            #asymmetric
            logits_2d=tf.reshape(logits_2d, [-1, tf.shape(x1d)[1], tf.shape(x1d)[1], 19, 19])
            #symmetric
            #logits_2d_sym=tf.transpose(logits_2d, perm=[0,2,1,4,3])
            #logits_2d = 0.5 * (logits_2d + logits_2d_sym)

            hot_encode = tf.one_hot(tf.cast(y1d, tf.int32), depth=19)
            hot_encode_e = tf.expand_dims(hot_encode, axis=2)
            #?, 500, 1, 20
            ones = tf.ones((1, tf.shape(x1d)[1]))
            #v0each row is the same; each row for all the positions and hot-encode for each position
            v0=tf.einsum('DieB,ej->DjiB', hot_encode_e, ones)
            v1=tf.expand_dims(v0, axis=-1)
            v2=tf.matmul(logits_2d, v1)
            v3=tf.squeeze(v2, axis=-1)
            v4=tf.transpose(v3, perm=[0,3,1,2])
            diag=tf.zeros_like(v4[:,:,:,0])
            v5=tf.matrix_set_diag(v4, tf.zeros_like(v4[:,:,:,0]))
            v6=tf.transpose(v5, perm=[0,2,3,1])

            mrf_2d_reduced=tf.reduce_sum(v6, axis=2)
            self.mrf_1d = logits_1d
            self.mrf_2d = logits_2d
            mrf_logits = tf.add(logits_1d, mrf_2d_reduced)
            mrf_logits = tf.pad(mrf_logits, [[0,0], [0,0], [0, 1]])

            debug=False
            if debug:
                self.left_1d=left_1d
                self.right_1d=right_1d
                self.hot_encode=hot_encode
                self.hot_encode_e=hot_encode_e
                self.ones=ones
                self.v0=v0
                self.v1=v1
                self.v2=v2
                self.v3=v3
                self.v4=v4
                self.v5=v5
                self.v6=v6
                self.diag=diag
            return mrf_logits

    def evaluate1d(self, mode):
        self.sess.run(self.iterator.initializer,\
                feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(mode)})
        acc = []
        total = 0
        while True:
            try:
                pred, y1d, size, name = self.sess.run([self.pred, self.y1d, self.size, self.name],
                        feed_dict={self.training:False})
                for y_, pred_, size_, name_ in zip(y1d, pred, size, name):
                    acc_ = util.calc_acc1d(pred_[:size_], y_[:size_])
                    acc.append(acc_)
                    total += 1
            except tf.errors.OutOfRangeError:
                break
        acc_ = np.mean(np.array(acc))
        logging.info('{:s} total= {} 1d_acc= {}'.format(mode, total, acc_))
        return

    def evaluate_mrf(self, mode, epoch):
        self.sess.run(self.iterator.initializer,\
                feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(mode)})
        acc = []
        total = 0

        save_dir = '{}/pred_{}'.format(self.train_config.out_pred_dir, epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        acc = []
        total = 0
        while True:
            try:
                pred, y1d, size, name, mrf_1d, mrf_2d =\
                        self.sess.run([self.pred, self.y1d, self.size, self.name, self.mrf_1d, self.mrf_2d])
                for y_, pred_, size_, name_, mrf_1d_, mrf_2d_ in zip(y1d, pred, size, name,
                        mrf_1d, mrf_2d):
                    save_path = '{}/{}.pred'.format(save_dir, name_)
                    np.savez(save_path, mrf_1d=mrf_1d_[:size_], mrf_2d=mrf_2d_[:size_, :size_], y=y_[:size_])
                    acc_ = util.calc_acc1d(pred_[:size_], y_[:size_])
                    acc.append(acc_)
                    total += 1
            except tf.errors.OutOfRangeError:
                break
        acc_ = np.mean(np.array(acc))
        logging.info('{:s} total= {} 1d_acc= {}'.format(mode, total, acc_))
        return

    def build_input(self):
        with tf.device('/cpu:0'):
            def parser(record):
                keys_to_features = {
                'x1d' :tf.FixedLenFeature([], tf.string),
                'x2d' :tf.FixedLenFeature([], tf.string),
                'y1d' :tf.FixedLenFeature([], tf.string),
                'y2d' :tf.FixedLenFeature([], tf.string),
                'size':tf.FixedLenFeature([], tf.int64),
                'name':tf.FixedLenFeature([], tf.string)
                }
                parsed = tf.parse_single_example(record, keys_to_features)
                size = parsed['size']
                name = parsed['name']
                x1d = tf.decode_raw(parsed['x1d'], tf.float32)
                x2d = tf.decode_raw(parsed['x2d'] ,tf.float32)
                y1d = tf.decode_raw(parsed['y1d'] ,tf.int16)
                y2d = tf.decode_raw(parsed['y2d'] ,tf.int16)
                x1d = tf.reshape(x1d, tf.stack([size, -1]))
                x2d = tf.reshape(x2d, tf.stack([size, size, -1]))
                y1d = tf.reshape(y1d, tf.stack([size, -1]))
                y2d = tf.reshape(y2d, tf.stack([size, size]))
                print 'shape', x1d.shape, x2d.shape, y2d.shape
                return x1d, x2d, y1d, y2d, size, name

            def filter_fn(x1d, x2d, y1d, y2d, size, name):
                return tf.size(y1d) <= PADDING_FULL_LEN

            dataset = tf.data.TFRecordDataset(self.input_tfrecord_files)
            dataset = dataset.map(parser, num_parallel_calls=8)
            dataset = dataset.shuffle(buffer_size=256)
            dataset = dataset.filter(filter_fn)
            batch_size=1
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(64)
            iterator = dataset.make_initializable_iterator()
            x1d, x2d, y1d, y2d, size, name = iterator.get_next()
            print 'check shape', x1d.shape, x2d.shape, y1d.shape
            print 'check shape', tf.shape(x1d), tf.shape(x2d)
            x1d=tf.reshape(x1d, [1, -1, self.x1d_channel_dim])
            y1d=tf.reshape(y1d, [1, -1])
            x2d=tf.reshape(x2d, [1, tf.shape(x1d)[1], tf.shape(x1d)[1], self.x2d_channel_dim])
            print 'check shape', x1d.shape, x2d.shape, y1d.shape
            return  x1d, x2d, y1d, y2d, size, name, iterator

    def train(self):
        self.x1d, self.x2d, self.y1d, self.y2d,\
                self.size, self.name, self.iterator = self.build_input()

        with tf.device('/gpu:{}'.format(self.train_config.gpu_label)):
            if self.train_config.model_type == 'full_resn1d':
                logits = self.resn1d(self.x1d)
                self.pred = tf.nn.softmax(logits)
                labels = tf.one_hot(tf.cast(self.y1d, tf.int32), depth=self.model_config['1d_label_size'])
            elif self.train_config.model_type == 'full_resn_mrf':
                logits = self.resn_mrf(self.x1d, self.x2d, self.y1d)
                self.pred = tf.nn.softmax(logits)
                labels = tf.one_hot(tf.cast(self.y1d, tf.int32), depth=self.model_config['1d_label_size'])

            log_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits))
            reg_loss = tf.losses.get_regularization_loss()
            self.loss = log_loss  + reg_loss
            if self.train_config.model_type == 'full_resn_mrf':
                mrf_1d_reg = tf.reduce_mean(tf.reduce_sum(tf.square(self.mrf_1d),axis=[1,2]))
                mrf_2d_reg = tf.reduce_mean(tf.reduce_sum(tf.square(self.mrf_2d),axis=[1,2,3,4]))
                mrf_reg_loss = self.train_config.mrf_1d_reg * mrf_1d_reg\
                        + self.train_config.mrf_2d_reg * mrf_2d_reg
                self.loss +=  mrf_reg_loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#for batch normalization
            with tf.control_dependencies(update_ops):
                if self.train_config.op_alg == 'adam':
                    optim = tf.train.AdamOptimizer(self.train_config.learn_rate,
                            beta1=self.train_config.beta1)#.minimize(self.loss)
                elif self.train_config.op_alg == 'sgd':
                    optim = tf.train.GradientDescentOptimizer(
                            self.train_config.learn_rate)#.minimize(self.loss)
            t_vars = tf.trainable_variables()
            accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False)\
                    for t_var in t_vars]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]
            batch_grads_vars = optim.compute_gradients(self.loss, t_vars)
            accum_ops = [accum_tvars[i].assign_add(batch_grad_var[0])
                    for i, batch_grad_var in enumerate(batch_grads_vars)]

            train_step = optim.apply_gradients([(accum_tvars[i]/self.train_config.batch_size,
                batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)])

        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('train_reg_loss', reg_loss)
        tf.summary.scalar('train_log_loss', log_loss)

        if self.train_config.model_type == 'full_resn_mrf':
            tf.summary.scalar('mrf_reg_loss', mrf_reg_loss)
            tf.summary.scalar('mrf_1d_reg', mrf_1d_reg)
            tf.summary.scalar('mrf_2d_reg', mrf_2d_reg)

        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.train_config.summary_dir, self.sess.graph)
        tf.global_variables_initializer().run()
        steps = 0
        saver = tf.train.Saver(max_to_keep=2000)

        debug=False
        if debug:
            input_tfrecords=self.dataset.get_chunks('train')
            #random.shuffle(input_tfrecords)
            self.sess.run(self.iterator.initializer,
                    feed_dict={self.input_tfrecord_files:input_tfrecords})
            left_1d, right_1d, logits_2d, hot_encode, hot_encode_e, ones, v0, v1, v2, v3, v4, v5, v6, diag, y1d=self.sess.run(
                    [self.left_1d, self.right_1d, self.mrf_2d, self.hot_encode, self.hot_encode_e, self.ones, self.v0, self.v1, self.v2, self.v3, self.v4, self.v5, self.v6, self.diag, self.y1d],
                    feed_dict={self.training:True})

        for epoch in np.arange(self.train_config.epoch):
            input_tfrecords=self.dataset.get_chunks('train')
            random.shuffle(input_tfrecords)
            self.sess.run(self.iterator.initializer,
                    feed_dict={self.input_tfrecord_files:input_tfrecords})

            train_loss = 0.0
            train_log_loss = 0.0
            train_reg_loss = 0.0
            train_mrf_reg_loss = 0.0
            n = 0
            while True:
                try:
                    self.sess.run(zero_ops)
                    for batch_index in range(self.train_config.batch_size):
                        if self.train_config.model_type == 'full_resn_mrf':
                            _, _loss, _log_loss, _reg_loss, _mrf_reg_loss, summary = self.sess.run(
                                    [accum_ops, self.loss, log_loss, reg_loss, mrf_reg_loss, merged_summary],
                                    feed_dict={self.training:True})
                            train_mrf_reg_loss += _mrf_reg_loss
                        else:
                            _, _loss, _log_loss, _reg_loss, summary = self.sess.run(
                                    [accum_ops, self.loss, log_loss, reg_loss, merged_summary],
                                    feed_dict={self.training:True})
                        train_loss += _loss
                        train_log_loss += _log_loss
                        train_reg_loss += _reg_loss
                        train_writer.add_summary(summary, steps)
                        steps += 1
                        n += 1
                    self.sess.run(train_step)
                except tf.errors.OutOfRangeError:
                    break
            saver.save(self.sess, '{}/model'.format(self.train_config.out_model_dir),
                    global_step=epoch)
            if self.train_config.model_type == 'full_resn_mrf':
                logging.info('Epoch= {:d} n= {} train loss= {:.4f} log_loss= {:.4f} reg_loss= {:.4f} mrf_reg_loss= {:.4f}'.format(
                    epoch, n, train_loss / n, train_log_loss/n, train_reg_loss/n, train_mrf_reg_loss/n))
            else:
                logging.info('Epoch= {:d} n= {} train loss= {:.4f} log_loss= {:.4f} reg_loss= {:.4f}'.format(
                    epoch, n, train_loss / n, train_log_loss/n, train_reg_loss/n))

            if self.train_config.model_type == 'full_resn1d':
                evaluate=self.evaluate1d
                self.evaluate1d('val')
                self.evaluate1d('casp12')
                self.evaluate1d('cameo')
            elif self.train_config.model_type == 'full_resn_mrf':
                self.evaluate_mrf('val', epoch)
                self.evaluate_mrf('casp12', epoch)
                self.evaluate_mrf('cameo', epoch)
        train_writer.close()

