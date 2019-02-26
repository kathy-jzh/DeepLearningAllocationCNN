import pandas as pd
import numpy as np
import tensorflow as tf
import os

from utils import load_pickle,log
from data.data_processing import DataHandler, get_training_data_from_path


# TODO see if we need to use a datahandler so that we will run the predictions

class Backtester:
    def __init__(self,path_data,path_model_to_restore):
        self._path_model_to_restore = path_model_to_restore
        self._name_network = 'CondensedGoogLeNet'
        self._path_data = path_data
        #we need to restore the output op and just to use this one to get the pred


        # self._handler = DataHandler() # Todo add param, actually we will not need it since we have everything in trainig all data


        self._df_all_data = None
        self._output = None
        self.__LOGGER_ENV = 'backtest'

    def run_backtest(self):
        X = self.get_df_all_data()
        pred = self.run_predictions(X)
        self._make_df_for_bckt(pred)

        self._df_all_data = self._df_all_data.sort_index() # by chronologic order
        df_backt_results = self._run_strategy()
        return df_backt_results ## TODO see if we keep it as an attribute

    def _run_strategy(self):
        df_data = self._df_all_data


    # use df_all_data and builds a df with dates and a return value (first value is one)

    def _make_df_for_bckt(self,pred):
        df_signals = pd.DataFrame(index=self._df_all_data.index,data=pred,columns=['long','hold','short'])
        self._df_all_data = pd.concat([self._df_all_data,df_signals],axis=1)


    def restore_output_op(self,sess):
        saver = tf.train.import_meta_graph(self._path_model_to_restore)
        folder = os.path.dirname(self._path_model_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(folder))

        graph = tf.get_default_graph()

        self._output = graph.get_tensor_by_name('{}/output:0'.format(self._name_network))
        self._x = graph.get_tensor_by_name('{}/x:0'.format(self._name_network))


    def run_predictions(self,X):
        sess = tf.Session()
        graph = tf.get_default_graph()
        self.restore_output_op(sess)
        pred = sess.run(self._output,feed_dict={self._x:X})
        sess.close()
        return pred
        # must run ouput and add the predictions in the dataframe df_all_data as 3 columns

    def get_df_all_data(self):
        samples_path = self._path_data
        # list of files in the folder: samples_paths
        list_file_names = os.listdir(samples_path)
        assert len(list_file_names) >= 1, 'The number of files in the folder {} is probably 0, it must be >=1'.format(
            samples_path)
        log('******* Getting data from folder: {}, Nb of files : {}, First file : {}'.format(samples_path,
                                                                                             len(list_file_names),
                                                                                             list_file_names[0]),self.__LOGGER_ENV)

        for i, file_name in enumerate(list_file_names):
            path = os.path.join(samples_path, file_name)
            df_all_data_one_batch = load_pickle(path, logger_env=self.__LOGGER_ENV)
            log('first_date: {}, last_date: {}'.format(df_all_data_one_batch.index[0], df_all_data_one_batch.index[-1]),
                environment=self.__LOGGER_ENV)
            # Keeping only the dates we want for the backtest
            df_all_data_one_batch = df_all_data_one_batch.loc[self._start_date:self._last_date]

            # in each of the pickle files the data is sorted in chronologic order we resort in case it is not
            df_all_data_one_batch = df_all_data_one_batch.sort_index()
            X = np.concatenate([[sample for sample in df_all_data_one_batch['sample'].values]],
                               axis=0)  # need this to get an array
            # Y = np.concatenate([[sample for sample in df_all_data_one_batch[targets_type].values]], axis=0)
            # n_samples = Y.shape[0]
            df_all_data_one_batch_for_bckt = df_all_data_one_batch[['PERMNO','PRC']]
            if i==0:
                X_res = X
                df_all_data = df_all_data_one_batch_for_bckt
            else:
                X_res = np.concatenate([X_res,X])
                df_all_data = pd.concat([df_all_data,df_all_data_one_batch_for_bckt])
        # now we must not change the order in X_res and df_all_data
        self._df_all_data = df_all_data
        return X_res



    def log(self,msg):
        log(msg,environment=self.__LOGGER_ENV)

