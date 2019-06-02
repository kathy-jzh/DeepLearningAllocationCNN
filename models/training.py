import numpy as np
import tensorflow as tf
import random
import time
from sklearn.metrics import confusion_matrix

from models.CNNModels import Net
from utils import log, extract_minibatch
from config.config import DEFAULT_LOG_ENV, NB_OF_EPOCHS_FOR_BAYESIAN


def train_model(X, Y, valX, valY,
                net: Net,
                model_ckpt_path,
                batch_size=128,
                epochs=50,
                save_step=10,
                display_step=1,
                restore=False,
                model_ckpt_path_to_restore=None,
                cost_sensitive_loss=False,
                cost_matrix=np.array([[0, 1.1, 1.3], [1.1, 0, 1.1], [1.3, 1.1, 0]]),
                is_bayesian=False,
                **kwargs):
    """
    :param X: samples, encoded images
    :param Y: one-hot-encoded labels
    :param valX: same as above used for validation
    :param valY: same as above used for validation
    :param net: Net object
    :param model_ckpt_path: path to save the checkpoints
    :param batch_size: size of one batch
    :param epochs: number of total epochs to train
    :param save_step: model will be saved every save_step epochs, set to 0 to save only at the end, set to -1 to never save
    :param display_step: display lags every display_step epochs
    :param restore: set to True to restore a previous model
    :param model_ckpt_path_to_restore:  path to use to restore a model checkpoint
    :param cost_sensitive_loss: boolean, will implement cost sensitive cross entropy loss function if true.
     See Kukar and Kononenko (1998) for details.
    :param cost_matrix: a matrix of size output_dimension by output_dimension, default to be identity. (i,j)-th element is a
                         penalty factor assigned for misclassifying i to j. It used to combine sample prior
                         (distribution of classes in the training sample) to compute expected penalty of misclassifying i-th class.
                         See Kukar and Kononenko (1998) for details
    :param is_bayesian: boolean, if true, will ensemble the result from the last NB_OF_EPOCHS_FOR_BAYESIAN epochs of training
    :return:
            1. prediction of validation data based on current model
            2. Training loss of current model
            3. validation loss of current model
    """

    # Get sample distribution to calculate expected penalty
    sample_size, _, image_size, channel_size = X.shape

    # number of batches
    nb_of_batches_training = max(int(sample_size / batch_size), 1)

    # calculate expected penalty if we consider a cost sensitive  loss
    if cost_sensitive_loss:
        sample_prior = Y.sum(axis=0) / len(Y)
        expected_penalty = np.dot(cost_matrix, sample_prior) / (1 - sample_prior)
    else:
        expected_penalty = None
    kwargs.update({'expected_penalty': expected_penalty})

    if restore:
        tf.reset_default_graph()
        sess = tf.Session()
        graph = tf.get_default_graph()
        net.restore_importants_ops(sess, model_ckpt_path_to_restore)
        return _run_epochs(restore, sess, net, X, Y, valX, valY, sample_size, epochs, nb_of_batches_training,
                           batch_size, display_step, save_step, model_ckpt_path, is_bayesian)

    else:
        with tf.Graph().as_default():
            with tf.variable_scope(net.name, reuse=None):
                net.build_operations(**kwargs)
                with tf.Session() as sess:
                    sess.run(net.init)
                    return _run_epochs(restore, sess, net, X, Y, valX, valY, sample_size, epochs,
                                       nb_of_batches_training, batch_size, display_step, save_step, model_ckpt_path,
                                       is_bayesian)


def _run_epochs(restore, sess, net, X, Y, valX, valY, sample_size, epochs,
                nb_of_batches_training, batch_size, display_step,
                save_step, model_ckpt_path, is_bayesian):
    """
    Runs the training, See train_model docstring for details
    """
    writer = tf.summary.FileWriter('logs', sess.graph)

    dropout = net.get_dropout()

    training_loss, val_loss, pred = [], [], []
    for e in range(epochs):
        # Display the epoch beginning
        if (e + 1) % display_step == 0:
            text_to_print = '\n***************** Epoch: {:03d}/{:03d} *****************'.format(e + 1,
                                                                                                epochs)
            log(text_to_print, DEFAULT_LOG_ENV)

        epoch_start_time = time.time()

        # Shuffle dataset before doing an epoch of training # todo see if we delete it
        Z = list(zip(X, Y))
        random.shuffle(Z)
        X, Y = zip(*Z)

        cc, aa = 0, 0
        for b in range(nb_of_batches_training):
            x_b, y_b = extract_minibatch(X, Y, batch_size=batch_size, current_batch=b)
            _, __, c, a = sess.run([net.optimizer, net.global_step, net.loss, net.accuracy],
                                   feed_dict={net.x: x_b, net.y: y_b, net.phase_train: True, net.dropout: dropout})
            # Make evaluation on a batch basis
            cc += (c * len(y_b))  # len(y_b) since the last batch can be of size smaller than batch size
            aa += (a * len(y_b))
            if (b + 1) % max(int(0.1 * nb_of_batches_training), 1) == 0:
                log('Batch {}/{} done ***  CumLoss'
                    ': {:.4f} CumAccuracy: {:.4f}'.format(b + 1, nb_of_batches_training,
                                                          cc / (batch_size * (b + 1)),
                                                          aa / (batch_size * (b + 1))), DEFAULT_LOG_ENV)

        epoch_loss, epoch_acc = cc / float(sample_size), aa / float(sample_size)
        training_loss.append(epoch_loss)

        # run validation in batches to avoid memory problems
        size_1_image = np.prod(valX[0].shape)
        limit_size = (10 * batch_size) * 42 * 42 * 5
        size_batch = int(limit_size / size_1_image) + 1
        output_val, epoch_val_loss = np.zeros((0, 3)), 0.
        nb_batches = round(len(X) / size_batch)
        log('To avoid killing the kernel, we run predictions in {} batches'.format(nb_batches),
            environment=DEFAULT_LOG_ENV)
        for batch in range(0, len(X), size_batch):
            X_batch, Y_batch = valX[batch:batch + size_batch], valY[batch:batch + size_batch]
            if X_batch.shape[0] == 0:
                break
            pred_batch, epoch_val_loss_batch = sess.run([net.output, net.loss],
                                                        feed_dict={net.x: X_batch, net.y: Y_batch,
                                                                   net.phase_train: False, net.dropout: 0.})
            output_val = np.concatenate([output_val, pred_batch])
            epoch_val_loss += epoch_val_loss_batch
        epoch_val_loss /= nb_batches

        conf_matrix = get_confusion_matrix(valY, output_val)
        rows_conf_matrix = ['{}\n'.format(row_i) for row_i in conf_matrix]
        log('Confusion Matrix : \n ' + ''.join(rows_conf_matrix), environment=DEFAULT_LOG_ENV)
        pred.append(output_val)
        val_loss.append(epoch_val_loss)
        epoch_val_acc = np.trace(conf_matrix) / np.sum(conf_matrix)

        # Displaying training info for the epoch
        if (e + 1) % display_step == 0:
            text_to_print = "Training time: {}  ======== Loss: {:.4f} Accuracy: {:.4f} val_acc: {:.4f} ".format(
                round(time.time() - epoch_start_time, 2), epoch_loss, epoch_acc, epoch_val_acc)
            log(text_to_print, DEFAULT_LOG_ENV)

        # Saving the model
        if save_step >= 1 and (e + 1) % save_step == 0:
            net.save(sess, model_ckpt_path, verbose=True, epoch=e + 1, write_meta_graph=(not restore))

        # Adding values to summary
        summary_str = sess.run(net.summary_op,
                               feed_dict={net.x: x_b, net.y: y_b, net.phase_train: True, net.dropout: dropout})
        writer.add_summary(summary_str, sess.run(net.global_step))
    if save_step >= 0:
        net.save(sess, model_ckpt_path, verbose=True, epoch=epochs, write_meta_graph=True)

    writer.close()

    if is_bayesian:
        # Sample from the last NB_OF_EPOCHS_FOR_BAYESIAN epochs to form Bayesian learning:
        return (np.mean(pred[-NB_OF_EPOCHS_FOR_BAYESIAN:], axis=0), training_loss, val_loss)
    else:
        return (pred[-1], training_loss, val_loss)


def get_confusion_matrix(ytrue, ypred):
    """
    Create confusion matrix based on true results ytrue and predicted result ypred
    """
    ytrue = np.argmax(ytrue, axis=1)
    ypred = np.argmax(ypred, axis=1)
    return confusion_matrix(ytrue, ypred)
