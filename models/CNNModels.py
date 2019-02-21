import tensorflow as tf
import os

from models.net import Net
from utils import log
from config.hyperparams import DEFAULT_LOG_ENV

# todo do functions for every block so that the graph is easier to read

class CondensedGoogLeNet(Net):
    """ A parametrically reduced version of GoogleNet-like CNN (For description and implementation of full GoogLeNet, see Szegedy et. al. (2015))

        This model preserves three layers of inception module. The realization is self-evident in the following codes.
        Hyperparameter lists for vanilla convolutional blocks and inception modules are defined in hyperparam_list.py

        """

    def __init__(self, shape_x:iter, shape_y:iter, hyperparams:dict,name='CondensedGoogLeNet'):
        super().__init__(shape_x, shape_y, hyperparams,name)

    def _inference(self,**kwargs):
        inputs = self.x
        try:
            dropout = self._hyperparams['dropout']
        except KeyError as e:
            log('dropout not found in hyperparams, dropout default will be 0.15',loglevel='warning',environment=DEFAULT_LOG_ENV)
            dropout = 0.15


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
        outputs = tf.nn.softmax(x,name='output')
        self._pre_thresholded_output = x
        self.output = outputs
        # return x, outputs

    def restore_importants_ops(self,sess,model_ckpt_path_to_restore):
        saver = tf.train.import_meta_graph(model_ckpt_path_to_restore)
        folder = os.path.dirname(model_ckpt_path_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(folder))

        graph = tf.get_default_graph()
        # print(graph.get_operations())

        self.x = graph.get_tensor_by_name('{}/x:0'.format(self.name))
        self.y = graph.get_tensor_by_name('{}/y:0'.format(self.name))

        self._pre_thresholded_output = graph.get_tensor_by_name('{}/dense_layer/BiasAdd:0'.format(self.name))
        self.output = graph.get_tensor_by_name('{}/output:0'.format(self.name))
        self.loss = graph.get_tensor_by_name('{}/loss/Mean:0'.format(self.name))

        self.global_step = graph.get_tensor_by_name('{}/global_step:0'.format(self.name))
        self.optimizer = graph.get_tensor_by_name('{}/minimize:0'.format(self.name))
        self.accuracy = graph.get_tensor_by_name('{}/accuracy/Mean:0'.format(self.name))
        self.summary_op = graph.get_tensor_by_name('{}/Merge/MergeSummary:0'.format(self.name))
        # self.init = graph.get_tensor_by_name('{}/init:0'.format(self.name))
        # self.init = tf.global_variables_initializer()
        self.saver = saver

# class ResNet:
#     def __init__(self):

# todo adapt this class to the new Net interface
class CondensedAlexNet(Net):
    """ A parametrically reduced version of AlexNet (For full version of AlexNet, see Krizhevsky (2012))

        Preserved the structure/architecture of AlexNet, but reduced in layers/hyperparameters

        """

    def __init__(self, num_of_classes: int):
        super().__init__(num_of_classes)

    def inference(self, inputs, hyperparams, **kwargs):
        try:
            dropout = kwargs['dropout']
        except KeyError as e:
            print('dropout not found in kwargs, dropout must be specified for inference method in CondensedAlexNet')
            raise e
        batch_size, _, image_size, channel_size = inputs.shape
        inputs = tf.reshape(inputs, [-1, image_size, image_size, channel_size])
        hyperparams = self._hyperparams

        # 1st convolutional layer block
        x = tf.nn.conv2d(inputs, filter=hyperparams["conv1_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME",
                         name="conv1_conv")
        x = self.add_bias(x)
        x = tf.nn.tanh(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="conv1_maxP")

        # 2nd convolutional layer block
        x = tf.nn.conv2d(x, filter=hyperparams["conv2_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME",
                         name="conv2_conv")
        x = self.add_bias(x)
        x = tf.nn.tanh(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="conv3_maxP")

        # 3rd convolutional layer block
        x = tf.nn.conv2d(x, filter=hyperparams["conv3_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME",
                         name="conv3_conv")
        x = self.add_bias(x)
        x = tf.nn.tanh(x)

        # 4th convolutional layer block
        x = tf.nn.conv2d(x, filter=hyperparams["conv4_conv_kernel"], strides=[1, 1, 1, 1], padding="SAME",
                         name="conv4_conv")
        x = self.add_bias(x)
        x = tf.nn.tanh(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID", name="conv4_maxP")

        # Flatten the inputs
        batch_size, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, [-1, h * w * c])
        x = tf.nn.dropout(x, keep_prob=1 - dropout)

        # 1st dense layer
        x = self.dense_layer(x, units=150, name={"weight": "dense1_weight", "bias": "dense1_bias"})
        x = tf.nn.tanh(x)

        # 2nd dense layer
        x = self.dense_layer(x, units=25, name={"weight": "dense2_weight", "bias": "dense2_bias"})
        x = tf.nn.tanh(x)

        pre_thresholded_output = self.dense_layer(x, units=self.num_of_classes,
                                                  name={"weight": "dense3_weight", "bias": "dense3_bias"})
        outputs = tf.nn.softmax(x)

        # Return the outputs
        return pre_thresholded_output, outputs

#
# def VAE_alexnet(inputs, hyperparams=VAE_hyperparam, dropout=0.2, hidden_dim=150):
#     batch_size, _, image_size, channel_size = inputs.shape
#     inputs = tf.reshape(inputs, [-1, image_size, image_size, channel_size])
#
#     # Encoding part
#     x = tf.nn.conv2d(inputs, filter=hyperparams['encoder_conv1_kernel'], strides=[1, 1, 1, 1], padding="SAME",
#                      name="encoder_conv1")
#     x = add_bias(x)
#     x = tf.nn.conv2d(x, filter=hyperparams['encoder_conv2_kernel'], strides=[1, 1, 1, 1], paddnig="SAME",
#                      name="encoder_conv2")
#     x = add_bias(x)
#
#     m, v = tf.nn.moments(x)
#     x = tf.nn.batch_normalization(x, mean=m, variance=v, scale=None, offset=None, variance_epsilon=1e-7)
#     x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="encoder_maxpool")
#     x = tf.nn.dropout(x, keep_prob=1 - dropout)
#     x = tf.layers.flatten(x)
#
#     hidden, mu, log_sigma2 = dense_layer(x, units=hidden_dim), dense_layer(x, units=hidden_dim), dense_layer(x,
#                                                                                                              units=hidden_dim)
#
#     # Decoding part
#     x = dense_layer(hidden, units=image_size * image_size * channel_size)
#     x = tf.nn.dropout(x, keep_prob=1 - dropout)
#     x = tf.reshape(x, [-1, image_size, image_size, channel_size])
#     x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="decoder_maxpool")
#     x = tf.nn.conv2d(inputs, filter=hyperparams['decoder_conv1_kernel'], strides=[1, 1, 1, 1], padding="SAME",
#                      name="encoder_conv1")
#     x = add_bias(x)
#     x = tf.nn.conv2d(x, filter=hyperparams['decoder_conv2_kernel'], strides=[1, 1, 1, 1], paddnig="SAME",
#                      name="encoder_conv2")
#     output = add_bias(x)
#
#     return output, hidden, mu, log_sigma2
