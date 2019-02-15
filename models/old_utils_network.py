import tensorflow as tf


def dropout(layer_name: str, inputs, keep_prob: float):
    # dropout_rate = 1 - keep_prob
    with tf.name_scope(layer_name):
        return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)


def concat(layer_name, inputs: iter):
    """
    concatenates a list of tensors on the channel exis
    :param layer_name:
    :param inputs:
    :return:
    """
    with tf.name_scope(layer_name):
        one_by_one = inputs[0]
        three_by_three = inputs[1]
        five_by_five = inputs[2]
        pooling = inputs[3]
        return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)


def conv2d(layer_name: str, inputs, out_channels: int, kernel_size: int, strides=1, padding='SAME'):
    """
    :param layer_name: for the scope
    :param inputs: input tensor (first one would be .x)
    :param out_channels: nb of out channels
    :param kernel_size: size of the mask
    :param strides: strides for the convolution
    :param padding: for the borders
    :return: output tensor of the convolutionnal layer
    """
    in_channels = inputs.get_shape()[-1]
    with tf.variable_scope(layer_name) as scope:
        # mask weights and biases
        w = tf.get_variable(name='weights',
                            trainable=True,
                            shape=[kernel_size, kernel_size, in_channels, out_channels],
                            # [filter_height, filter_width, in_channels, out_channels]
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',
                            trainable=True,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
        inputs = tf.nn.bias_add(inputs, b, name='bias_add')
        outputs = tf.nn.relu(inputs, name='relu')  # TODO see if we keep this activation function ?
        return outputs


def max_pool(layer_name: str, inputs, pool_size: int, strides: int = 1, padding='SAME'):
    # TODO see if we modify the padding so that it reduces dimension ?
    """
    :param layer_name: for the scope
    :param inputs: input 4-D tensor (first one would be the output of the first convolution)
    :param strides: strides for the pooling
    :param padding: for the borders either 'VALID' or 'SAME'
    :return: output tensor
    """
    with tf.name_scope(layer_name):
        return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                              name=layer_name)


def avg_pool(layer_name: str, inputs, pool_size: int, strides: int = 1, padding='SAME'):
    # TODO see if we modify the padding so that it reduces dimension ?
    """
    :param layer_name: for the scope
    :param inputs: input 4-D tensor
    :param strides: strides for the pooling
    :param padding: for the borders either 'VALID' or 'SAME'
    :return: output tensor of the convolutionnal layer
    """
    with tf.name_scope(layer_name):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                              name=layer_name)
