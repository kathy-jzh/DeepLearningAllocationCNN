import tensorflow as tf
import numpy as np

from config import Config


# TODO: see later how to pass efficiently config arguments
# TODO: replace the 'print' by an efficent logger

class Net:
    def __init__(self,_cfg:Config):
        self.config = _cfg
        self.saver = None
        # init the global step
        self._init_global_step()
        # init the epoch counter
        self._init_cur_epoch()
        self.scope = {}

    # save function thet save the checkpoint in the path defined in config
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiments path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def _init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def _init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self): # todo useful ?
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError('Should be implemented in the child class')

    def build_model(self):
        raise NotImplementedError('Should be implemented in the child class')

    # TODO: understand what this does
    def load_with_skip(self, data_path, session, skip_layer):
        data_dict = np.load(data_path, encoding='latin1').item()  # type: dict
        for key in data_dict.keys():
            if key not in skip_layer:
                # with tf.variable_scope(key, reuse=True, auxiliary_name_scope=False):
                with tf.variable_scope(self.scope[key], reuse=True) as scope:
                    with tf.name_scope(scope.original_name_scope):
                        for subkey, data in data_dict[key].items():
                            session.run(tf.get_variable(subkey).assign(data))
