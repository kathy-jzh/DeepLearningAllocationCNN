import tensorflow as tf
import os

from models.net import Net


######## 3 Classes, one for each type of neural network:

# - CondensedGoogLeNet
# - CondensedAlexNet
# - ResNet


class CondensedGoogLeNet(Net):
    """ A parametrically reduced version of GoogleNet-like CNN (For description and implementation of full GoogLeNet, see Szegedy et. al. (2015))

        This model preserves three layers of inception module. The realization is self-evident in the following codes.
        Hyperparameter dictionnary is in config.py

        All
        """

    def __init__(self, shape_x: iter, shape_y: iter, hyperparams: dict, name='CondensedGoogLeNet'):
        super().__init__(shape_x, shape_y, hyperparams, name)

    def _inference(self, **kwargs):
        inputs = self.x
        dropout = self.dropout

        hyperparams_first_blocks = self._hyperparams['first_block']
        inception1_hyperparam = self._hyperparams['inception_1']
        inception2_hyperparam = self._hyperparams['inception_2']
        inception3_hyperparam = self._hyperparams['inception_3']

        # 1st convolutional layer block
        x = self.conv2d(layer_name='conv1_conv', inputs=inputs,
                        kernel_shape=hyperparams_first_blocks['conv1_conv_kernel'], strides=1, padding='SAME',
                        activation_func=tf.nn.tanh)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='conv1_maxP')

        # 2nd convolutional layer block
        x = self.conv2d(layer_name='conv2_conv', inputs=x,
                        kernel_shape=hyperparams_first_blocks['conv2_conv_kernel'], strides=1, padding='SAME',
                        activation_func=tf.nn.tanh)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='conv2_maxP')

        # 1st inception module
        x = self.inception_mod(x, inception1_hyperparam, name='inception1')
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='inception1_maxP')

        # 2nd inception module
        x = self.inception_mod(x, inception2_hyperparam, name='inception2')
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='inception2_maxP')

        # 3rd inception module
        x = self.inception_mod(x, inception3_hyperparam, name='inception3')
        x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID", name='inception3_avgP')

        # Flatten and dropout
        x = tf.layers.flatten(x)
        x = tf.nn.dropout(x, keep_prob=1 - dropout)

        x = self.dense_layer(x, units=self._shape_y[-1], name='dense_layer')
        self._pre_thresholded_output = x
        outputs = tf.nn.softmax(x, name='output')
        self.output = outputs

    @staticmethod
    def inception_mod(inputs, hyperparams: dict, name: str):
        """
        Well-tuned inception module used in the subsequent GoogLeNet-like CNN
        :param inputs: tensor
        :param hyperparams: dictionnary of hyperparameters for the sizes of the kernels
        :param name: for variable scope
        """
        with tf.variable_scope(name):
            # 1x1 pathway
            x1 = Net.conv2d(layer_name='1x1_conv', inputs=inputs, kernel_shape=hyperparams["1x1_conv_kernel"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME')

            # 1x1 to 3x3 pathway
            x2 = Net.conv2d(layer_name='3x3_conv1', inputs=inputs, kernel_shape=hyperparams["3x3_conv_kernel2"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME',
                            kernel_shape_pre=hyperparams["3x3_conv_kernel1"])

            # 1x1 to 5x5 pathway
            x3 = Net.conv2d(layer_name='5x5_conv1', inputs=inputs, kernel_shape=hyperparams["5x5_conv_kernel2"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME',
                            kernel_shape_pre=hyperparams["5x5_conv_kernel1"])

            # 3x3 to 1x1 pathway
            x4 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name="pooling1")

            x4 = Net.conv2d(layer_name='pooling1_conv', inputs=x4, kernel_shape=hyperparams["pooling1_conv_kernel"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME')

            x = tf.concat([x1, x2, x3, x4], axis=3)  # Concat in the 4th dim to stack
            outputs = tf.tanh(x)

            return outputs

    def restore_importants_ops(self, sess, model_ckpt_path_to_restore):
        """
        Used when we do not need to build all the network but only to restore a few operations to continue the training
        :param sess: current tf session
        :param model_ckpt_path_to_restore: path for the model's checkpoint file to restore
        """
        saver = tf.train.import_meta_graph(model_ckpt_path_to_restore)
        folder = os.path.dirname(model_ckpt_path_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(folder))

        graph = tf.get_default_graph()

        self.x = graph.get_tensor_by_name('{}/x:0'.format(self.name))
        self.y = graph.get_tensor_by_name('{}/y:0'.format(self.name))
        self.phase_train = graph.get_tensor_by_name('{}/phase_train:0'.format(self.name))
        self.dropout = graph.get_tensor_by_name('{}/dropout:0'.format(self.name))

        self._pre_thresholded_output = graph.get_tensor_by_name('{}/dense_layer/BiasAdd:0'.format(self.name))
        self.output = graph.get_tensor_by_name('{}/output:0'.format(self.name))
        self.loss = graph.get_tensor_by_name('{}/loss/Mean:0'.format(self.name))

        self.global_step = graph.get_tensor_by_name('{}/global_step:0'.format(self.name))
        self.optimizer = graph.get_tensor_by_name('{}/minimize:0'.format(self.name))
        self.accuracy = graph.get_tensor_by_name('{}/accuracy/Mean:0'.format(self.name))
        self.summary_op = graph.get_tensor_by_name('{}/Merge/MergeSummary:0'.format(self.name))

        self.saver = saver


class CondensedAlexNet(Net):
    """ A parametrically reduced version of AlexNet (For full version of AlexNet, see Krizhevsky (2012))

        Preserved the structure/architecture of AlexNet, but reduced in layers/hyperparameters

        """

    def __init__(self, shape_x: iter, shape_y: iter, hyperparams: dict, name: str = 'CondensedAlexNet'):
        super().__init__(shape_x, shape_y, hyperparams, name)

    def _inference(self, **kwargs):
        dropout = self.dropout
        inputs = self.x
        hyperparams = self._hyperparams

        # 1st convolutional layer block
        x = self.conv2d(inputs=inputs, kernel_shape=hyperparams["conv1_conv_kernel"], strides=1, padding="SAME",
                        layer_name="conv1_conv", activation_func=tf.nn.tanh)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="conv1_maxP")

        # 2nd convolutional layer block
        x = self.conv2d(inputs=x, kernel_shape=hyperparams["conv2_conv_kernel"], strides=1, padding="SAME",
                        layer_name="conv2_conv", activation_func=tf.nn.tanh)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="conv3_maxP")

        # 3rd convolutional layer block
        x = self.conv2d(inputs=x, kernel_shape=hyperparams["conv3_conv_kernel"], strides=1, padding="SAME",
                        layer_name="conv3_conv", activation_func=tf.nn.tanh)

        # 4th convolutional layer block
        x = self.conv2d(inputs=x, kernel_shape=hyperparams["conv4_conv_kernel"], strides=1, padding="SAME",
                        layer_name="conv4_conv", activation_func=tf.nn.tanh)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID", name="conv4_maxP")

        # Flatten the inputs
        batch_size, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, [-1, h * w * c])
        x = tf.nn.dropout(x, rate=dropout)

        # 1st dense layer
        x = self.dense_layer(x, units=150, name='dense_layer_1')
        x = tf.nn.tanh(x)

        # 2nd dense layer
        x = self.dense_layer(x, units=25, name='dense_layer_2')
        x = tf.nn.tanh(x)

        x = self.dense_layer(x, units=self._shape_y[-1], name='dense_layer_3')
        self._pre_thresholded_output = x
        self.output = tf.nn.softmax(x, name='output')

    def restore_importants_ops(self, sess, model_ckpt_path_to_restore):
        """
        Used when we do not need to build all the network but only to restore a few operations to continue the training
        :param sess: current tf session
        :param model_ckpt_path_to_restore: path for the model's checkpoint file to restore
        """
        saver = tf.train.import_meta_graph(model_ckpt_path_to_restore)
        folder = os.path.dirname(model_ckpt_path_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(folder))

        graph = tf.get_default_graph()

        self.x = graph.get_tensor_by_name('{}/x:0'.format(self.name))
        self.y = graph.get_tensor_by_name('{}/y:0'.format(self.name))
        self.phase_train = graph.get_tensor_by_name('{}/phase_train:0'.format(self.name))
        self.dropout = graph.get_tensor_by_name('{}/dropout:0'.format(self.name))

        self._pre_thresholded_output = graph.get_tensor_by_name('{}/dense_layer_3/BiasAdd:0'.format(self.name))
        self.output = graph.get_tensor_by_name('{}/output:0'.format(self.name))
        self.loss = graph.get_tensor_by_name('{}/loss/Mean:0'.format(self.name))

        self.global_step = graph.get_tensor_by_name('{}/global_step:0'.format(self.name))
        self.optimizer = graph.get_tensor_by_name('{}/minimize:0'.format(self.name))
        self.accuracy = graph.get_tensor_by_name('{}/accuracy/Mean:0'.format(self.name))
        self.summary_op = graph.get_tensor_by_name('{}/Merge/MergeSummary:0'.format(self.name))

        self.saver = saver


class ResNet(Net):
    def __init__(self, shape_x: iter, shape_y: iter, hyperparams: dict, num_filters: int = 10,
                 block_sizes=None, name='ResNet'):
        """
        :param shape_x: shape of samples
        :param shape_y: shape of labels
        :param hyperparams: hyperparams as a dictionnary
        :param num_filters: The number of filters to use for the first block layer
                of the model. This number is then doubled for each subsequent block
                layer.
        :param block_sizes: A list containing n values, where n is the number of sets of
                block layers desired. Each value should be the number of blocks in the
                i-th set.
        :param name: name of the model
        """
        super().__init__(shape_x, shape_y, hyperparams, name)
        if block_sizes is None:
            block_sizes = [3, 3, 3, 2]
        self._num_filters = num_filters  # nb of channels after the first convolution
        self._block_sizes = block_sizes

        self._bottleneck = False
        self._block_strides = [1 for i in block_sizes]

    def _inference(self, **kwargs):
        inputs = self.x
        training = self.phase_train

        x = self.conv2d(layer_name='initial_conv', inputs=inputs,
                        kernel_shape=(3, 3, 5, self._num_filters), strides=2, padding='SAME',
                        activation_func=tf.identity)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name='initial_max_pool')

        for i, num_blocks in enumerate(self._block_sizes):
            num_filters = self._num_filters * (2 ** i)
            x = self.__block_layer(
                inputs=x, filters=num_filters, bottleneck=self._bottleneck, blocks=num_blocks,
                strides=self._block_strides[i], training=training,
                name='block_layer{}'.format(i + 1))

        x = self.batch_normalization(x, training)
        x = tf.nn.relu(x)

        # ResNet does an Average Pooling layer over pool_size,
        # but that is the same as doing a reduce_mean. We do a reduce_mean
        # here because it performs better than AveragePooling2D.
        x = tf.reduce_mean(input_tensor=x, axis=[1, 2], keepdims=True)
        x = tf.identity(x, 'final_reduce_mean')

        x = tf.squeeze(x, [1, 2])

        x = self.dense_layer(x, units=self._shape_y[-1], name='dense_layer')
        self._pre_thresholded_output = x
        outputs = tf.nn.softmax(x, name='output')
        self.output = outputs

    def restore_importants_ops(self, sess, model_ckpt_path_to_restore):
        """
        Used when we do not need to build all the network but only to restore a few operations to continue the training
        :param sess: current tf session
        :param model_ckpt_path_to_restore: path for the model's checkpoint file to restore
        """
        saver = tf.train.import_meta_graph(model_ckpt_path_to_restore)
        folder = os.path.dirname(model_ckpt_path_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(folder))

        graph = tf.get_default_graph()

        self.x = graph.get_tensor_by_name('{}/x:0'.format(self.name))
        self.y = graph.get_tensor_by_name('{}/y:0'.format(self.name))
        self.phase_train = graph.get_tensor_by_name('{}/phase_train:0'.format(self.name))
        self.dropout = graph.get_tensor_by_name('{}/dropout:0'.format(self.name))

        self._pre_thresholded_output = graph.get_tensor_by_name('{}/dense_layer/BiasAdd:0'.format(self.name))
        self.output = graph.get_tensor_by_name('{}/output:0'.format(self.name))
        self.loss = graph.get_tensor_by_name('{}/loss/Mean:0'.format(self.name))

        self.global_step = graph.get_tensor_by_name('{}/global_step:0'.format(self.name))
        self.optimizer = graph.get_tensor_by_name('{}/minimize:0'.format(self.name))
        self.accuracy = graph.get_tensor_by_name('{}/accuracy/Mean:0'.format(self.name))
        self.summary_op = graph.get_tensor_by_name('{}/Merge/MergeSummary:0'.format(self.name))

        self.saver = saver

    @staticmethod
    def __block_layer(inputs, filters, bottleneck, blocks, strides, training, name):
        """
        Creates one layer of blocks for the ResNet model.

        :param inputs: A tensor of size[batch, height_in, width_in, channels]
        :param filters:  The number of filters for the first convolution of the layer.
        :param bottleneck: Is the block created a bottleneck block.
        :param blocks:  The number of blocks contained in the layer.
        :param strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        :param training:  Either True or False, whether we are currently training the
            model. Needed for batch norm.
        :param name: A string name for the tensor output of the block layer.
        :return: The output tensor of the block layer.
        """

        with tf.variable_scope(name):
            # Bottleneck blocks end with 4x the number of filters as they start with
            filters_out = filters * 4 if bottleneck else filters

            def projection_shortcut(inputs):
                return Net.conv2d(inputs=inputs, kernel_shape=(1, 1, inputs.shape[-1], filters_out), strides=strides,
                                  padding='SAME',
                                  activation_func=tf.identity, layer_name='projection')

            # Only the first block per block_layer uses projection_shortcut and strides
            inputs = ResNet._building_block(inputs, filters, training, projection_shortcut, strides, name='block_0')

            for i in range(1, blocks):
                inputs = ResNet._building_block(inputs, filters, training, None, 1, name='block_' + str(i))

            return tf.identity(inputs, name)

    @staticmethod
    def _building_block(inputs, filters, training, projection_shortcut, strides, name: str):
        """
        :param inputs: A tensor of size[batch, height_in, width_in, channels]
        :param filters:  The number of filters for the first convolution of the layer.
        :param training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
        :param projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        :param strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        :param name: A string name for the tensor output of the block layer.
        :return: The output tensor of the block; shape should match inputs.
        """
        with tf.variable_scope(name):
            shortcut = inputs
            inputs = Net.batch_normalization(inputs, training)
            inputs = tf.nn.relu(inputs)

            # The projection shortcut should come after the first batch norm and ReLU
            # since it performs a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            inputs = Net.conv2d(inputs=inputs, kernel_shape=(3, 3, inputs.shape[-1], filters), strides=strides,
                                padding='SAME',
                                activation_func=tf.identity, layer_name='conv1_building_block')

            inputs = Net.batch_normalization(inputs, training)
            inputs = tf.nn.relu(inputs)
            inputs = Net.conv2d(inputs=inputs, kernel_shape=(3, 3, inputs.shape[-1], filters), strides=1,
                                padding='SAME',
                                activation_func=tf.identity, layer_name='conv2_building_block')
            return inputs + shortcut
