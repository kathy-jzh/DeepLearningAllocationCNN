import tensorflow as tf
from utils import log
from config.hyperparams import DEFAULT_LEARNING_RATE, DEFAULT_TF_OPTIMIZER,DEFAULT_LOG_ENV


class Net:
    """
    a mother class for all types of nets
    """

    def __init__(self, shape_x: iter, shape_y: iter, hyperparams: dict,name:str):
        self._hyperparams = hyperparams
        self.name = name


        self._shape_x = shape_x
        self._shape_y = shape_y
        self._pre_thresholded_output = None

        self.x = None
        self.y = None
        self.init = None
        self.output = None
        self.global_step = None
        self.optimizer = None
        self.loss = None
        self.accuracy = None

        self.summary_op = None

        self.saver = None


    def _inference(self, **kwargs):
        """

        instanciate: pre_thresholded_output, output
        """
        raise NotImplementedError('Should be implemented in the child class')

    @staticmethod
    def dense_layer(inputs, units, name=None):
        """
        A wrapped up dense layer function (defines dense layer operation)
        :param inputs:
        :param units:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            inputs_shape = tf.Tensor.get_shape(inputs).as_list()[-1]

            # weight_matrix = tf.Variable(tf.truncated_normal([inputs_shape, units], stddev=0.01), name=name['weight'])
            # bias = tf.Variable(tf.constant(1.0, shape=[units]), name=name['bias'])

            w = tf.get_variable(name='weight',
                                trainable=True,
                                shape=[inputs_shape, units],
                                # [filter_height, filter_width, in_channels, out_channels]
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable(name='bias',
                                trainable=True,
                                shape=[units],
                                initializer=tf.constant_initializer(0.0))

            outputs = tf.matmul(inputs, w)
            outputs = tf.nn.bias_add(outputs, b)
            return outputs

    @staticmethod
    def add_bias(inputs):
        """
        A lambda function to return proper size of bias added to conv layer
        :param inputs:
        :return:
        """
        return tf.nn.bias_add(inputs, tf.Variable(tf.constant(0.0, shape=[tf.Tensor.get_shape(inputs).as_list()[-1]])))

    @staticmethod
    def inception_mod(inputs, hyperparams: dict, name: str):
        """
        Well-tuned inception module used in the subsequent GoogLeNet-like CNN
        :param inputs:
        :param hyperparams:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 1x1 pathway
            # x1 = tf.nn.conv2d(inputs, filter=hyperparams["1x1_conv_kernel"], strides=[1, 1, 1, 1], padding='SAME',
            #                   name="1x1_conv")
            # x1 = Net.add_bias(x1)
            # x1 = tf.nn.tanh(x1)
            x1 = Net.conv2d(layer_name='1x1_conv', inputs=inputs, kernel_shape=hyperparams["1x1_conv_kernel"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME')

            # 1x1 to 3x3 pathway
            # x2 = tf.nn.conv2d(inputs, filter=hyperparams["3x3_conv_kernel1"], strides=[1, 1, 1, 1], padding='SAME',
            #                   name="3x3_conv1")
            # x2 = tf.nn.conv2d(x2, filter=hyperparams["3x3_conv_kernel2"], strides=[1, 1, 1, 1], padding='SAME',
            #                   name="3x3_conv2")
            # x2 = Net.add_bias(x2)
            # x2 = tf.nn.tanh(x2)
            x2 = Net.conv2d(layer_name='3x3_conv1', inputs=inputs, kernel_shape=hyperparams["3x3_conv_kernel2"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME',
                            kernel_shape_pre=hyperparams["3x3_conv_kernel1"])

            # 1x1 to 5x5 pathway
            # x3 = tf.nn.conv2d(inputs, filter=hyperparams["5x5_conv_kernel1"], strides=[1, 1, 1, 1], padding='SAME',
            #                   name="5x5_conv1")
            # x3 = tf.nn.conv2d(x3, filter=hyperparams["5x5_conv_kernel2"], strides=[1, 1, 1, 1], padding='SAME',
            #                   name="5x5_conv2")
            # x3 = Net.add_bias(x3)
            # x3 = tf.nn.tanh(x3)
            x3 = Net.conv2d(layer_name='5x5_conv1', inputs=inputs, kernel_shape=hyperparams["5x5_conv_kernel2"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME',
                            kernel_shape_pre=hyperparams["5x5_conv_kernel1"])

            # 3x3 to 1x1 pathway
            x4 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name="pooling1")
            # x4 = tf.nn.conv2d(x4, filter=hyperparams['pooling1_conv_kernel'], strides=[1, 1, 1, 1], padding='SAME',
            #                   name="pooling1_conv")
            # x4 = Net.add_bias(x4)
            # x4 = tf.nn.tanh(x4)
            x4 = Net.conv2d(layer_name='pooling1_conv', inputs=inputs, kernel_shape=hyperparams["pooling1_conv_kernel"],
                            strides=1, activation_func=tf.nn.tanh, padding='SAME')

            x = tf.concat([x1, x2, x3, x4], axis=3)  # Concat in the 4th dim to stack
            outputs = tf.tanh(x)

            return outputs

    @staticmethod
    def conv2d(layer_name: str, inputs, kernel_shape: iter, strides=1, padding='SAME',
               activation_func=tf.nn.tanh, kernel_shape_pre=None):
        """
        :param layer_name: for the scope
        :param inputs: input tensor (first one would be .x)
        :param out_channels: nb of out channels
        :param kernel_size: size of the mask
        :param strides: strides for the convolution
        :param padding: for the borders
        :return: output tensor of the convolutionnal layer
        """
        # in_channels = inputs.get_shape()[-1]

        with tf.variable_scope(layer_name):
            if kernel_shape_pre:
                with tf.variable_scope('pre_convolution'):
                    # mask weights and biases
                    w_pre = tf.get_variable(name='weights_pre',
                                            trainable=True,
                                            shape=kernel_shape_pre,
                                            # [filter_height, filter_width, in_channels, out_channels]
                                            initializer=tf.contrib.layers.xavier_initializer())
                    inputs = tf.nn.conv2d(inputs, filter=w_pre, strides=[1, strides, strides, 1], padding=padding,
                                          name='conv')
            with tf.variable_scope('convolution'):
                # mask weights and biases
                w = tf.get_variable(name='weights',
                                    trainable=True,
                                    shape=kernel_shape,
                                    # [filter_height, filter_width, in_channels, out_channels]
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name='biases',
                                    trainable=True,
                                    shape=[kernel_shape[-1]],
                                    initializer=tf.constant_initializer(0.0))
                inputs = tf.nn.conv2d(inputs, filter=w, strides=[1, strides, strides, 1], padding=padding, name='conv')
                inputs = tf.nn.bias_add(inputs, b, name='bias_add')
                outputs = activation_func(inputs,
                                          name='activation_func')  # TODO see if we keep this activation function ?
                return outputs

    @staticmethod
    def __cost_sensitive_loss_func(y_true, y_pred, expected_penalty):
        """
        Cost sensitive loss function with cross-entropy
        :param y_true:
        :param y_pred:
        :param expected_penalty:
        :return:
        """
        y_pred = tf.clip_by_value \
            ((y_pred * expected_penalty) / tf.reduce_sum(y_pred * expected_penalty, axis=-1, keepdims=True), 1e-7,
             1 - 1e-7)
        loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss)

        #    def vae_loss(X, X_output, mu, log_sigma2):
        #        reconstruction_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=X, predictions=X_output))
        #        distribution_loss = -0.5*tf.reduce_mean(-tf.exp(log_sigma2) - tf.square(mu) + log_sigma2, axis=-1)
        #        return tf.reduce_mean(reconstruction_loss+distribution_loss)

    def _xentropy_loss_func(self, cost_sensitive_loss=False, expected_penalty=None, **kwargs):
        """

        :param output: if cost_sensitive_loss is False output should be the pre_thresholded_output
        :param y_batch: y_batch
        :param expected_penalty: expected penalty of misclassifying i-th class.
        :return: loss function
        """
        y_batch = self.y
        with tf.name_scope('loss'):
            if cost_sensitive_loss:
                assert expected_penalty is not None, 'expected penalty in x_entropy_loss_function must not be None when cost_sensitive_loss = True'
                xentropy_loss = self.__cost_sensitive_loss_func(y_batch, self.output, expected_penalty)
            else:
                xentropy_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=self._pre_thresholded_output))
            self.loss = xentropy_loss

    def _optimize(self):

        try:
            learning_rate = self._hyperparams['learning_rate']
        except KeyError:
            log('learning_rate was not specified in hyperparams using default:{}'.format(DEFAULT_LEARNING_RATE),
                loglevel='warning',environment=DEFAULT_LOG_ENV)
            learning_rate = DEFAULT_LEARNING_RATE  # todo define default rate in config

        try:
            tf_optimizer = self._hyperparams['tf_optimizer']
        except KeyError:
            log('tf_optimizer was not specified in hyperparams, using default:{}'.format(DEFAULT_TF_OPTIMIZER),
                loglevel='warning',environment=DEFAULT_LOG_ENV)
            tf_optimizer = DEFAULT_TF_OPTIMIZER
        if tf_optimizer.lower() == 'adam':
            tf_optimizer = tf.train.AdamOptimizer
        elif tf_optimizer.lower() == 'sgd':
            tf_optimizer = tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError('You need to modify the code to use another optimizer so far we have sgd and adam')

        tf.summary.scalar('training_loss', self.loss)
        self.optimizer = tf_optimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step,name='minimize')
        # todo find how the learning rate changes result


    def save(self, sess:tf.Session,model_ckpt_path:str,verbose=False,epoch=None,write_meta_graph=True):

        if verbose:
            msg = 'Saving Model to {}, global step = {}'.format(model_ckpt_path,self.global_step.eval(sess))
            msg = msg +' epoch: {}'.format(epoch) if epoch else msg
            log(msg,environment=DEFAULT_LOG_ENV)
        self.saver.save(sess, model_ckpt_path, self.global_step,write_meta_graph=write_meta_graph)
        if verbose:
            log('Model Saved', environment=DEFAULT_LOG_ENV)

    def _accuracy_func(self):
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1)), tf.float32))
            tf.summary.scalar("validation_error", (1.0 - accuracy))
            self.accuracy = accuracy

    def build_operations(self, **kwargs):
        """
        Builds all operations that will be run within the session
        :param kwargs: union (as a dict) of all arguments necessary for the functions below
        :return:
        """
        self.x = tf.placeholder(tf.float32, self._shape_x, name='x')
        self.y = tf.placeholder(tf.float32, self._shape_y, name='y')
        self._inference(**kwargs)  # example: dropout
        self._xentropy_loss_func(**kwargs)  # example: expected_penalty

        self.global_step = tf.Variable(0, trainable=False)
        self._optimize()  # todo see if we can do in a different way for the optimizer
        self._accuracy_func()
        self.summary_op = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

        max_save_to_keep = kwargs.get('max_save_to_keep', 4)
        self.saver =  tf.train.Saver(max_to_keep=kwargs.get('max_save_to_keep',4))
        log('Saver will keep the last {} latest models'.format(max_save_to_keep),DEFAULT_LOG_ENV)


