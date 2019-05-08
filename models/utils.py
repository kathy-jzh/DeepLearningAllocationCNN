import tensorflow as tf
import random
import numpy as np



class UtilsTraining:

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