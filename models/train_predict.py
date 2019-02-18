import numpy as np
import tensorflow as tf
import random
import time
from sklearn.metrics import confusion_matrix

from models.CNNModels import Net
from utils import log
from models.utils import UtilsTraining
from config.hyperparams import DEFAULT_LOG_ENV, NB_OF_EPOCHS_FOR_BAYESIAN


def train_predict(X, Y, valX, valY,
                  net: Net, model_ckpt_path, cost_sensitive_loss=False,
                  cost_matrix=np.array([[0, 1.1, 1.3], [1.1, 0, 1.1], [1.3, 1.1, 0]]),
                  is_bayesian=False,
                  batch_size=128, epochs=50, save_step=10, display_step=1,
                  restore=False,model_ckpt_path_to_restore=None, **kwargs):
    """ Function to train and make prediction with models

        Args:
            X, Y, valX, valY: training set (X, Y), validation set (valX, valY). The function will return a prediction based
                              on the trained model on validation set valX
            net: specified model created with TensorFlow. In this research, will take either Condensed_AlexNet or
                   Condensed_Googlenet, object of the Net class
            model_ckpt_path: path to save the model checkpoint
            cost_sensitive_loss: boolean, will implement cost sensitive cross entropy loss function if true. See Kukar and Kononenko (1998) for details.
            cost_matrix: a matrix of size output_dimension by output_dimension, default to be identity. (i,j)-th element is a
                         penalty factor assigned for misclassifying i to j. It used to combine sample prior
                         (distribution of classes in the training sample) to compute expected penalty of misclassifying i-th class.
                         See Kukar and Kononenko (1998) for details
            is_bayesian: boolean, if true, will ensemble the result from the last NB_OF_EPOCHS_FOR_BAYESIAN epochs of training
            batch_size, epochs,: some paramters for training the model
            *args, **kwargs: arguments for functional input model. See arguments for the 3 types of models.

        Return:
            1. prediction of validation data based on current model
            2. Training loss of current model
            3. validation loss of current model

        Will save the session to prespecified model checkpoint path for reuse.
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

        sess = tf.Session()
        graph = tf.get_default_graph()
        net.restore_importants_ops(sess, model_ckpt_path_to_restore)


        # sess.run(net.init)
        writer = tf.summary.FileWriter('logs', sess.graph)

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

            # X,Y = np.asarray(X),np.asarray(Y) # todo see when necessary

            cc, aa = 0, 0
            for b in range(nb_of_batches_training):
                x_b, y_b = UtilsTraining._extract_minibatch(X, Y, batch_size=batch_size, current_batch=b)
                sess.run([net.optimizer, net.global_step], feed_dict={net.x: x_b, net.y: y_b})

                # Make evaluation on a batch basis
                c, a = sess.run([net.loss, net.accuracy], feed_dict={net.x: x_b, net.y: y_b})
                cc += (c * len(y_b))  # len(y_b) since the last batch can be of size smaller than batch size
                aa += (a * len(y_b))
                if (b + 1) % max(int(0.1 * nb_of_batches_training), 1) == 0:
                    log('Batch {}/{} done ***  CumLoss'
                        ': {:.4f} CumAccuracy: {:.4f}'.format(b + 1, nb_of_batches_training,
                                                              cc / (batch_size * (b + 1)),
                                                              aa / (batch_size * (b + 1))), DEFAULT_LOG_ENV)
            epoch_loss, epoch_acc = cc / float(sample_size), aa / float(sample_size)
            training_loss.append(epoch_loss)

            output_val, epoch_val_loss, epoch_val_acc = sess.run([net.output, net.loss, net.accuracy],
                                                                 feed_dict={net.x: valX, net.y: valY})
            pred.append(output_val)
            val_loss.append(epoch_val_loss)

            # Displaying training info for the epoch
            if (e + 1) % display_step == 0:
                text_to_print = "Training time: {}  ======== Loss: {:.4f} Accuracy: {:.4f} val_Loss: {:.4f} val_acc: {:.4f} ".format(
                    round(time.time() - epoch_start_time, 2), epoch_loss, epoch_acc, epoch_val_loss,
                    epoch_val_acc)
                log(text_to_print, DEFAULT_LOG_ENV)

            # Saving the model
            if (e + 1) % save_step == 0:
                net.save(sess, model_ckpt_path, verbose=True, epoch=e + 1, write_meta_graph=True)

            # Adding values to summary
            # todo should'nt we consider valX and valY here ?
            summary_str = sess.run(net.summary_op, feed_dict={net.x: x_b, net.y: y_b})
            writer.add_summary(summary_str, sess.run(net.global_step))

        net.save(sess, model_ckpt_path, verbose=True, epoch=epochs, write_meta_graph=True)

        writer.close()

        if is_bayesian:
            # Sample from the last NB_OF_EPOCHS_FOR_BAYESIAN epochs to form Bayesian learning: average across last 10 epochs of predictions
            return (np.mean(pred[-NB_OF_EPOCHS_FOR_BAYESIAN:], axis=0), training_loss, val_loss)
        else:
            return (pred[-1], training_loss, val_loss)
    else:

        with tf.Graph().as_default():
            with tf.variable_scope(net.name,reuse=None):
                net.build_operations(**kwargs)
                with tf.Session() as sess:

                    sess.run(net.init)
                    writer = tf.summary.FileWriter('logs', sess.graph)

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

                        # X,Y = np.asarray(X),np.asarray(Y) # todo se when necessary

                        cc, aa = 0, 0
                        for b in range(nb_of_batches_training):
                            x_b, y_b = UtilsTraining._extract_minibatch(X, Y, batch_size=batch_size, current_batch=b)
                            sess.run([net.optimizer, net.global_step], feed_dict={net.x: x_b, net.y: y_b})

                            # Make evaluation on a batch basis
                            c, a = sess.run([net.loss, net.accuracy], feed_dict={net.x: x_b, net.y: y_b})
                            cc += (c * len(y_b))  # len(y_b) since the last batch can be of size smaller than batch size
                            aa += (a * len(y_b))
                            if (b + 1) % max(int(0.1 * nb_of_batches_training), 1) == 0:
                                log('Batch {}/{} done ***  CumLoss'
                                    ': {:.4f} CumAccuracy: {:.4f}'.format(b + 1, nb_of_batches_training,
                                                                          cc / (batch_size * (b + 1)),
                                                                          aa / (batch_size * (b + 1))), DEFAULT_LOG_ENV)
                        epoch_loss, epoch_acc = cc / float(sample_size), aa / float(sample_size)
                        training_loss.append(epoch_loss)

                        output_val, epoch_val_loss, epoch_val_acc = sess.run([net.output, net.loss, net.accuracy],
                                                                             feed_dict={net.x: valX, net.y: valY})
                        pred.append(output_val)
                        val_loss.append(epoch_val_loss)

                        # Displaying training info for the epoch
                        if (e + 1) % display_step == 0:
                            text_to_print = "Training time: {}  ======== Loss: {:.4f} Accuracy: {:.4f} val_Loss: {:.4f} val_acc: {:.4f} ".format(
                                round(time.time() - epoch_start_time, 2), epoch_loss, epoch_acc, epoch_val_loss,
                                epoch_val_acc)
                            log(text_to_print, DEFAULT_LOG_ENV)

                        # Saving the model
                        if (e + 1) % save_step == 0:
                            net.save(sess, model_ckpt_path, verbose=True, epoch=e + 1, write_meta_graph=True)

                        # Adding values to summary
                        # todo should'nt we consider valX and valY here ?
                        summary_str = sess.run(net.summary_op, feed_dict={net.x: x_b, net.y: y_b})
                        writer.add_summary(summary_str, sess.run(net.global_step))

                    net.save(sess, model_ckpt_path, verbose=True, epoch=epochs, write_meta_graph=True)

                    writer.close()

                    if is_bayesian:
                        # Sample from the last NB_OF_EPOCHS_FOR_BAYESIAN epochs to form Bayesian learning: average across last 10 epochs of predictions
                        return (np.mean(pred[-NB_OF_EPOCHS_FOR_BAYESIAN:], axis=0), training_loss, val_loss)
                    else:
                        return (pred[-1], training_loss, val_loss)

    # # graph = tf.Graph().as_default() if not restore else tf.get_default_graph()
    # graph = tf.Graph().as_default()
    # with graph:
    #     with tf.variable_scope(net.name):
    #         with sess:
    #             # Builds the operations (output, train_op, accuracy... ) or restores them
    #             # if restore:
    #             #     # model_ckpt_path_to_restore = model_ckpt_path_to_restore or
    #             #     assert model_ckpt_path_to_restore is not None,'model_ckpt_path_to_restore must not be None if restore is True'
    #             #     net.restore_importants_ops(model_ckpt_path_to_restore)
    #             # else:
    #             net.build_operations(**kwargs)
    #             if not restore:
    #                 sess.run(net.init)
    #             writer = tf.summary.FileWriter('logs', sess.graph)
    #
    #             training_loss, val_loss, pred = [], [], []
    #             for e in range(epochs):
    #                 # Display the epoch beginning
    #                 if (e + 1) % display_step == 0:
    #                     text_to_print = '\n***************** Epoch: {:03d}/{:03d} *****************'.format(e + 1, epochs)
    #                     log(text_to_print, DEFAULT_LOG_ENV)
    #
    #                 epoch_start_time = time.time()
    #
    #                 # Shuffle dataset before doing an epoch of training # todo see if we delete it
    #                 Z = list(zip(X, Y))
    #                 random.shuffle(Z)
    #                 X, Y = zip(*Z)
    #
    #                 # X,Y = np.asarray(X),np.asarray(Y) # todo se when necessary
    #
    #                 cc, aa = 0, 0
    #                 for b in range(nb_of_batches_training):
    #                     x_b, y_b = UtilsTraining._extract_minibatch(X, Y, batch_size=batch_size, current_batch=b)
    #                     sess.run([net.optimizer, net.global_step], feed_dict={net.x: x_b, net.y: y_b})
    #
    #                     # Make evaluation on a batch basis
    #                     c, a = sess.run([net.loss, net.accuracy], feed_dict={net.x: x_b, net.y: y_b})
    #                     cc += (c * len(y_b))  # len(y_b) since the last batch can be of size smaller than batch size
    #                     aa += (a * len(y_b))
    #                     if (b + 1) % max(int(0.1 * nb_of_batches_training), 1) == 0:
    #                         log('Batch {}/{} done ***  CumLoss'
    #                             ': {:.4f} CumAccuracy: {:.4f}'.format(b + 1, nb_of_batches_training,
    #                                                                   cc / (batch_size * (b + 1)),
    #                                                                   aa / (batch_size * (b + 1))), DEFAULT_LOG_ENV)
    #                 epoch_loss, epoch_acc = cc / float(sample_size), aa / float(sample_size)
    #                 training_loss.append(epoch_loss)
    #
    #                 output_val, epoch_val_loss, epoch_val_acc = sess.run([net.output, net.loss, net.accuracy],
    #                                                                      feed_dict={net.x: valX, net.y: valY})
    #                 pred.append(output_val)
    #                 val_loss.append(epoch_val_loss)
    #
    #                 # Displaying training info for the epoch
    #                 if (e + 1) % display_step == 0:
    #                     text_to_print = "Training time: {}  ======== Loss: {:.4f} Accuracy: {:.4f} val_Loss: {:.4f} val_acc: {:.4f} ".format(
    #                         round(time.time() - epoch_start_time, 2), epoch_loss, epoch_acc, epoch_val_loss,
    #                         epoch_val_acc)
    #                     log(text_to_print, DEFAULT_LOG_ENV)
    #
    #                 # Saving the model
    #                 if (e + 1) % save_step == 0:
    #                     net.save(sess, model_ckpt_path, verbose=True, epoch=e + 1,write_meta_graph=True)
    #
    #                 # Adding values to summary
    #                 # todo should'nt we consider valX and valY here ?
    #                 summary_str = sess.run(net.summary_op, feed_dict={net.x: x_b, net.y: y_b})
    #                 writer.add_summary(summary_str, sess.run(net.global_step))
    #
    #             net.save(sess, model_ckpt_path, verbose=True, epoch=epochs + 1, write_meta_graph=True)
    #
    #             writer.close()
    #
    #             if is_bayesian:
    #                 # Sample from the last NB_OF_EPOCHS_FOR_BAYESIAN epochs to form Bayesian learning: average across last 10 epochs of predictions
    #                 return (np.mean(pred[-NB_OF_EPOCHS_FOR_BAYESIAN:], axis=0), training_loss, val_loss)
    #             else:
    #                 return (pred[-1], training_loss, val_loss)


# todo do a function only for predictions ?

def get_confusion_matrix(ytrue, ypred):
    # Create confusion matrix based on true results ytrue and predicted result ypred

    ytrue = np.argmax(ytrue, axis=1)
    ypred = np.argmax(ypred, axis=1)
    return confusion_matrix(ytrue, ypred)


