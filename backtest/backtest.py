import pandas as pd
import numpy as np
import tensorflow as tf
import os
import ezhc as hc

from utils import load_pickle, log,plot_highstock_with_table
from data.data_processing import DataHandler, get_training_data_from_path


# TODO see if we need to use a datahandler so that we will run the predictions

class Backtester:
    def __init__(self, path_data, path_model_to_restore, start_date=20150101, end_date=20200101):
        self._path_model_to_restore = path_model_to_restore
        self._name_network = 'CondensedGoogLeNet'
        self._path_data = path_data

        self._start_date = start_date
        self._end_date = end_date
        # we need to restore the output op and just to use this one to get the pred


        self._df_all_data = None
        self._output = None
        self._x = None

        self._df_permnos_to_buy = None
        self.__LOGGER_ENV = 'backtest'

    def run_backtest(self):
        X = self.get_df_all_data()
        pred = self.run_predictions(X)
        self._make_df_for_bckt(pred)

        df_backt_results = self._run_strategy()

        df_strats = df_backt_results.astype(np.float64)
        df_strats.index.name='date'
        df_strats['Cash'] = 1.03**(5./252.)
        df_strats = df_strats.reset_index()
        df_strats.date = pd.to_datetime(df_strats.date.apply(lambda x: '{}-{}-{}'.format(str(x)[:4],str(x)[4:6],str(x)[6:])))
        df_strats = df_strats.set_index('date')
        self.df_strats = df_strats.cumprod()



    def _run_strategy(self):
        """ Uses the dataframe with all data: Example below, to make a backtest
               PERMNO     PRC      long      hold     short
        date
        20181030  83387.0    8.89  0.149458  0.353090  0.497452
        20181030  84207.0  119.15  0.404111  0.385122  0.210767
        20181030  83815.0   16.59  0.613428  0.268158  0.118414
        20181030  83835.0   55.55  0.317590  0.389825  0.292584
        20181030  83469.0   37.80  0.521703  0.326894  0.151403
        :return:


        """
        # TODO modify the function to support different strategies, make a column weights in df_permnos_to_buy
        df_data = self._df_all_data
        strategies = ['10_max_long', '20_max_long','2_max_long']
        self._df_permnos_to_buy = self.__create_signals(df_data,strategies=strategies)
        self._df_permnos_to_buy = self._df_permnos_to_buy.shift(1).dropna()

        # Compute Returns
        df_prices = df_data[['PRC', 'PERMNO']].pivot_table(columns='PERMNO', index='date')
        df_rets = df_prices / df_prices.shift(1)
        df_rets = df_rets.dropna().PRC

        # Run the backtest with the returns and the predictions
        self.log('Running Backtest')
        df_results = pd.DataFrame(columns=strategies,index=self._df_permnos_to_buy.index)
        for date in self._df_permnos_to_buy.index:
            for strat in strategies:
                list_permnos_to_buy = self._df_permnos_to_buy.loc[date][strat]
                df_results.loc[date][strat] = df_rets.loc[date][list_permnos_to_buy].mean()
        return df_results

    def __create_signals(self,df_data,strategies=['10_max_long','20_max_long']):
        """
        :param df_data: Example
                PERMNO     PRC      long      hold     short
        date
        20181030  83387.0    8.89  0.149458  0.353090  0.497452
        20181030  84207.0  119.15  0.404111  0.385122  0.210767
        20181030  83815.0   16.59  0.613428  0.268158  0.118414

        :return:   dataframe with the permnos to buy for each date # TODO extend it with an extra column weights
                                             permnos
        20181120  [85567.0, 83486.0, 83728.0, 83630.0, 85285.0, ...
        20181220  [85539.0, 83762.0, 83844.0, 83885.0, 83532.0, ...
        """
        self.log('Creating signals with strategies {}'.format(strategies))

        df_permnos_to_buy = pd.DataFrame(columns=strategies, index=sorted(set(df_data.index)))
        for date in df_data.index:
            for strat in strategies:
                data_sorted = df_data[['long', 'PERMNO']].reset_index().set_index(['date', 'PERMNO']).loc[date].sort_values(
                            by='long', ascending=False)
                if strat=='10_max_long':
                    list_permons_to_buy = list(data_sorted.index[:10])
                    df_permnos_to_buy.loc[date][strat] = list_permons_to_buy
                elif strat=='20_max_long':
                    list_permons_to_buy = list(data_sorted.index[:20])
                    df_permnos_to_buy.loc[date][strat] = list_permons_to_buy
                elif strat=='2_max_long':
                    list_permons_to_buy = list(data_sorted.index[:2])
                    df_permnos_to_buy.loc[date][strat] = list_permons_to_buy
                else:
                    raise NotImplementedError('The strategy {} is not implemented'.format(strat))
        self.log('Signals created')
        return df_permnos_to_buy
    # use df_all_data and builds a df with dates and a return value (first value is one)

    def _make_df_for_bckt(self, pred):
        """
        Instantiates self._df_all_data with a Dataframe containing prices and predictions and sorts it
        chronologically
                * Example:

                PERMNO     PRC      long      hold     short
        date
        20181030  83387.0    8.89  0.149458  0.353090  0.497452
        20181030  84207.0  119.15  0.404111  0.385122  0.210767
        20181030  83815.0   16.59  0.613428  0.268158  0.118414
        20181030  83835.0   55.55  0.317590  0.389825  0.292584
        20181030  83469.0   37.80  0.521703  0.326894  0.151403

        :param pred: numpy array shape (N_samples,3)
        :return: Nothing
        """
        self.log('Joining prices data and predictions')
        df_signals = pd.DataFrame(index=self._df_all_data.index, data=pred, columns=['long', 'hold', 'short'])
        self._df_all_data = pd.concat([self._df_all_data, df_signals], axis=1)
        self._df_all_data = self._df_all_data.sort_index()

    def restore_output_op(self, sess):
        """
        Restores tensors : x and output and instanciates self._x and self._output
        :param sess: tf.Session
        :return: Nothing
        """
        saver = tf.train.import_meta_graph(self._path_model_to_restore)
        folder = os.path.dirname(self._path_model_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(folder))

        graph = tf.get_default_graph()

        self._output = graph.get_tensor_by_name('{}/output:0'.format(self._name_network))
        self._x = graph.get_tensor_by_name('{}/x:0'.format(self._name_network))

    def run_predictions(self, X):
        """
        Restores a model saved with tensorflow and predicts signals
        :param X: numpy array shape (N_samples,16,16,4) (if we consider 16 pixels and 4 channels)
        :return: predictions as a numpy array shape (N_samples,3)
        """
        self.log('Restoring model and making the predictions')
        sess = tf.Session()
        graph = tf.get_default_graph()
        self.restore_output_op(sess)
        self.log('Model Restored, launching output operation')

        size_1_image = np.prod(X[0].shape)
        limit_size = 20*16*16*4
        size_batch = int(limit_size/size_1_image)+1
        pred = np.zeros((0,3))
        self.log('To avoid killing the kernel, we run predictions in {} batches'.format(round(len(X)/size_batch)))
        for batch in range(0,len(X),size_batch):
            X_batch = X[batch:batch+size_batch]
            pred_batch = sess.run(self._output, feed_dict={self._x: X_batch})
            pred = np.concatenate([pred,pred_batch])

        self.log('Predictions computed')
        sess.close()
        return pred
        # must run ouput and add the predictions in the dataframe df_all_data as 3 columns

    def plot_backtest(self):

        return plot_highstock_with_table(self.df_strats,title='Backtest for different strategies')


    def get_df_all_data(self):
        samples_path = self._path_data
        # list of files in the folder: samples_paths
        list_file_names = os.listdir(samples_path)
        assert len(list_file_names) >= 1, 'The number of files in the folder {} is probably 0, it must be >=1'.format(
            samples_path)
        log('******* Getting data from folder: {}, Nb of files : {}, First file : {}'.format(samples_path,
                                                                                             len(list_file_names),
                                                                                             list_file_names[0]),
            self.__LOGGER_ENV)

        for i, file_name in enumerate(list_file_names):
            path = os.path.join(samples_path, file_name)
            df_all_data_one_batch = load_pickle(path, logger_env=self.__LOGGER_ENV)
            self.log('first_date: {}, last_date: {}'.format(df_all_data_one_batch.index[0], df_all_data_one_batch.index[-1]))
            # Keeping only the dates we want for the backtest
            df_all_data_one_batch = df_all_data_one_batch.loc[self._start_date:self._end_date]
            self.log('new first_date: {}, new last_date: {}'.format(df_all_data_one_batch.index[0], df_all_data_one_batch.index[-1]))

            # in each of the pickle files the data is sorted in chronologic order we resort in case it is not
            df_all_data_one_batch = df_all_data_one_batch.sort_index()
            X = np.concatenate([[sample for sample in df_all_data_one_batch['sample'].values]],
                               axis=0)  # need this to get an array
            # Y = np.concatenate([[sample for sample in df_all_data_one_batch[targets_type].values]], axis=0)
            # n_samples = Y.shape[0]
            df_all_data_one_batch_for_bckt = df_all_data_one_batch[['PERMNO', 'PRC']]
            if i == 0:
                X_res = X
                df_all_data = df_all_data_one_batch_for_bckt
            else:
                X_res = np.concatenate([X_res, X])
                df_all_data = pd.concat([df_all_data, df_all_data_one_batch_for_bckt])
        # now we must not change the order in X_res and df_all_data
        self._df_all_data = df_all_data
        return X_res

    def log(self, msg):
        log(msg, environment=self.__LOGGER_ENV)
