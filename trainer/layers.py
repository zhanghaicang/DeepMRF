import tensorflow as tf
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training):
    return inputs
    #return tf.layers.batch_normalization(
    #        inputs=inputs, training=training)
    '''
    return tf.layers.batch_normalization(
            inputs=inputs, momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=training, fused=True, trainable=False)
    '''

def resn1d_regular_block_v1(inputs, act, filters_1d, kernel_size_1d,
        kernel_initializer, bias_initializer,
        kernel_regularizer, bias_regularizer,
        training, names):
    
    with tf.variable_scope(names) as scope:
        shortcut = inputs
        if inputs.shape[-1] != filters_1d:
            shortcut = tf.layers.conv1d(inputs=inputs, filters=filters_1d,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
            shortcut = batch_norm(shortcut, training=training)

        conv_1d = tf.layers.conv1d(inputs=inputs, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
        conv_1d = batch_norm(conv_1d, training=training)
        conv_1d = act(conv_1d)

        conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
        conv_1d = batch_norm(conv_1d, training=training)
        conv_1d = shortcut + conv_1d
        conv_1d = act(conv_1d)

        return conv_1d


def resn1d_regular_block_v2(inputs, act, filters_1d, kernel_size_1d,
        kernel_initializer, bias_initializer,
        kernel_regularizer, bias_regularizer,
        training, names):
    
    with tf.variable_scope(names) as scope:
        shortcut = inputs
        conv_1d = batch_norm(inputs, training=training)
        conv_1d = act(conv_1d)

        if inputs.shape[-1] != filters_1d:
            shortcut = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

        conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

        conv_1d = batch_norm(conv_1d, training=training)
        conv_1d = act(conv_1d)
        conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

        return shortcut + conv_1d

def resn1d_regular_block_v3(inputs, act, filters_1d, kernel_size_1d,
        kernel_initializer, bias_initializer,
        kernel_regularizer, bias_regularizer,
        training, names):
    
    with tf.variable_scope(names) as scope:
        shortcut = inputs
        if inputs.shape[-1] != filters_1d:
            shortcut = tf.layers.conv1d(inputs=inputs, filters=filters_1d,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)
        conv_1d = batch_norm(inputs, training=training)
        conv_1d = act(conv_1d)
        conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)

        conv_1d = batch_norm(conv_1d, training=training)
        conv_1d = act(conv_1d)
        conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

        return shortcut + conv_1d

def resn2d_regular_block(inputs, act, filters_2d, kernel_size_2d,
        kernel_initializer, bias_initializer,
        kernel_regularizer, bias_regularizer,
        training, names):
    
    with tf.variable_scope(names) as scope:
        shortcut = inputs
        if inputs.shape[-1] != filters_2d:
            shortcut = tf.layers.conv2d(inputs=inputs, filters=filters_2d,
                    kernel_size=kernel_size_2d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)
        conv_2d = batch_norm(inputs, training=training)
        conv_2d = act(conv_2d)
        conv_2d = tf.layers.conv2d(inputs=conv_2d, filters=filters_2d,
                kernel_size=kernel_size_2d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

        conv_2d = batch_norm(conv_2d, training=training)
        conv_2d = act(conv_2d)
        conv_2d = tf.layers.conv2d(inputs=conv_2d, filters=filters_2d,
                kernel_size=kernel_size_2d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

        return shortcut + conv_2d


def resn1d_bottleneck_block(inputs, act, filters_1d, kernel_size_1d,
        kernel_initializer, bias_initializer,
        kernel_regularizer, bias_regularizer,
        training):
    shortcut = inputs
    conv_1d = batch_norm(inputs, training=training)
    conv_1d = act(conv_1d)

    if inputs.shape[-1] == filters_1d:
        shortcut = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

    conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
            kernel_size=1, strides=1, padding='same',
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

    conv_1d = batch_norm(conv_1d, training=training)
    conv_1d = act(conv_1d)
    conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d/4,
            kernel_size=kernel_size_1d, strides=1, padding='same',
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

    conv_1d = batch_norm(conv_1d, training=training)
    conv_1d = act(conv_1d)
    
    conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
            kernel_size=1, strides=1, padding='same',
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

    return shortcut + conv_1d
