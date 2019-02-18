import tensorflow as tf
import random
import numpy as np



class UtilsTraining:

    # @staticmethod
    # def _cost_sensitive_loss_func(y_true, y_pred, expected_penalty):
    #     """
    #     Cost sensitive loss function with cross-entropy
    #     :param y_true:
    #     :param y_pred:
    #     :param expected_penalty:
    #     :return:
    #     """
    #     y_pred = tf.clip_by_value \
    #         ((y_pred * expected_penalty) / tf.reduce_sum(y_pred * expected_penalty, axis=-1, keepdims=True), 1e-7,
    #          1 - 1e-7)
    #     loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
    #     return tf.reduce_mean(loss)
    #
    #     #    def vae_loss(X, X_output, mu, log_sigma2):
    #     #        reconstruction_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=X, predictions=X_output))
    #     #        distribution_loss = -0.5*tf.reduce_mean(-tf.exp(log_sigma2) - tf.square(mu) + log_sigma2, axis=-1)
    #     #        return tf.reduce_mean(reconstruction_loss+distribution_loss)
    #
    # @staticmethod
    # def xentropy_loss_func(output, y_batch, cost_sensitive_loss=False, expected_penalty=None):
    #     """
    #
    #     :param output: if cost_sensitive_loss is False output should be the pre_thresholded_output
    #     :param y_batch: y_batch
    #     :param expected_penalty: expected penalty of misclassifying i-th class.
    #     :return: loss function
    #     """
    #     with tf.name_scope('loss'):
    #         if cost_sensitive_loss:
    #             assert expected_penalty is not None, 'expected penalty in x_entropy_loss_function must not be None when cost_sensitive_loss = True'
    #             xentropy_loss = UtilsTraining._cost_sensitive_loss_func(y_batch, output, expected_penalty)
    #         else:
    #             pre_thresholded_output = output
    #             xentropy_loss = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=pre_thresholded_output))
    #         return xentropy_loss
    #
    # @staticmethod
    # def accuracy_func(output, y_batch):
    #     with tf.name_scope('accuracy'):
    #         accuracy = tf.reduce_mean(
    #             tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y_batch, axis=1)), tf.float32))
    #         tf.summary.scalar("validation_error", (1.0 - accuracy))
    #         return accuracy
    #
    # @staticmethod
    # def optimizer(loss, global_step, learning_rate, tf_optimizer=tf.train.AdamOptimizer):
    #     tf.summary.scalar("cost", loss)
    #     optimizer = tf_optimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    #     # todo find how the learning rate changes result
    #     return optimizer

    @staticmethod
    def _extract_minibatch(X, Y, batch_size, current_batch):

        if (current_batch + 1) * batch_size > len(X):
            x_b, y_b = X[current_batch * batch_size:], Y[current_batch * batch_size:]

        else:
            x_b, y_b = X[current_batch * batch_size:(current_batch + 1) * batch_size], Y[current_batch * batch_size:(current_batch + 1) * batch_size]
        # if len(x_b)==0:
        #     print(X)
        #     print(X.shape)
        #     print(batch_size, current_batch, x_b)
        if len(x_b)==0: # todo need to change
            x_b,y_b = X[:batch_size],Y[:batch_size]
        return np.asarray(x_b), np.asarray(y_b)

    @staticmethod
    def next_batch(X, Y, batch_size, current_batch):
        Z = list(zip(X, Y))
        random.shuffle(Z)
        X, Y = zip(*Z)

        if (current_batch + 1) * batch_size > len(X):
            x_b, y_b = X[current_batch * batch_size:], Y[current_batch * batch_size:]
        else:
            x_b, y_b = X[current_batch * batch_size:(current_batch + 1) * batch_size], Y[current_batch * batch_size:(
                                                                                                                            current_batch + 1) * batch_size]
        return x_b, y_b