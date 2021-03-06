import pandas as pd
import numpy as np
import tensorflow as tf
import os

from utils import load_pickle, log, integer_to_timestamp_date_index


class Backtester:
    """
    Backtester:
        - gets data from pickles files
        - restores a trained network from a given checkpoint
        - makes predictions
        - creates 'signals' i.e what to buy at each rebalancing date for a list of strategies
        - runs the backtests for each strategies given market returns and signals
        - plots the backtesting results

    See main function Backtester.run_backtest()
    """

    def __init__(self, path_data, path_model_to_restore, start_date=20150101, end_date=20200101,
                 strategies=['10_max_long', '20_max_long', '2_max_long'], num_bins=None,
                 network_name='CondensedGoogLeNet'):
        """
        :param path_data:str: path to the folder containing the pickles files with the images
        :param path_model_to_restore:str: path for the model checkpoint used to make the predictions
        :param start_date:int: first date to consider for the backtest (format yyyymmdd)
        :param end_date:int:  last date to consider for the backtest
        :param strategies:list: list of strategies to backtest (see Backtester.__create_signals for supported names)
        :param network_name:str: name of the network to restore (used when restoring tensors)
        :param num_bins:int: number of bins in case we want to backtest a family of strategies
        """
        self._path_model_to_restore = path_model_to_restore
        self._name_network = network_name

        self._path_data = path_data
        self._start_date = start_date
        self._end_date = end_date

        self._strategies = strategies
        self._num_bins = num_bins

        # tensors that will be restored from checkpoint
        self._output = None
        self._x = None
        self._dropout = None
        self._phase_train = None

        # data retrieved or generated by the backtester
        self._df_all_data = None
        self._df_permnos_to_buy = None
        self.df_strats = None

        self.__LOGGER_ENV = 'backtest'

    def run_backtest(self):
        """
        central function of the backtester, instanciates self.df_strats containing backtest_results for each strategy
        """
        X = self.get_df_all_data()
        pred = self.run_predictions(X)
        self._make_df_for_bckt(pred)

        df_backt_results = self._run_strategies()

        self.df_strats = self._format_df_strats(df_backt_results)

    def _run_strategies(self):
        """ Uses the dataframe with all data to make a backtesting
        : Example below
               PERMNO     RET     long      hold     short
        date
        20181030  83387.0   1.03  0.149458  0.353090  0.497452
        20181030  84207.0   1.03  0.404111  0.385122  0.210767

        return a dataframe with weighted returns
        :return:                  1_decile_long 2_decile_long
                    20151001        0.92554       0.933354
                    20151008        1.00058        0.99515

        """
        df_data = self._df_all_data
        strategies = self._strategies
        self._df_permnos_to_buy = self.__create_signals(df_data, strategies=self._strategies, bins=self._num_bins)
        # self._df_permnos_to_buy = self._df_permnos_to_buy.shift(1).dropna() # because if we buy at i we get returns in i+1, not here
        # In the data returns at date i are the returns we would get if we buy at i

        df_rets = df_data[['RET', 'PERMNO']].pivot_table(columns='PERMNO', index='date')
        df_rets = df_rets.dropna().RET

        # Run the backtest with the returns and the predictions
        self.log('Running Backtest')
        df_results = pd.DataFrame(columns=strategies, index=self._df_permnos_to_buy.index)
        for date in self._df_permnos_to_buy.index:
            for strat in strategies:
                list_permnos_to_buy, weight_permnos_to_buy = self._df_permnos_to_buy.loc[date][strat]
                if len(list_permnos_to_buy) == 0:
                    df_results.loc[date][strat] = 1.  # we buy nothing
                else:
                    df_results.loc[date][strat] = np.sum(
                        np.array(df_rets.loc[date][list_permnos_to_buy]) * np.array(weight_permnos_to_buy))
        return df_results

    def __create_signals(self, df_data, strategies=['10_max_long', '20_max_long'], bins=None, equi_weighted=True):
        """
        Creates the list of stocks to buy in each strategy for each rebalacing date

        :param df_data: Example
                PERMNO     PRC      long      hold     short
        date
        20181030  83387.0    8.89  0.149458  0.353090  0.497452
        20181030  84207.0  119.15  0.404111  0.385122  0.210767
        :param strategies: list of strategies to consider

        :return:   dataframe1 with the permnos to buy for each date and the associated weights

        # df_permnos_to_buy
                                           4_max_long
        20181120  [85567.0, 83486.0, 83728.0, 83630.0],[0.42,0.08 ...]
        20181220  [85539.0, 83762.0, 83844.0, 83885.0],[0.05,0.048 ...]


        """
        self.log('Creating signals with strategies {}'.format(strategies))

        # we sort the predictions
        data_sorted_long_all_dates = df_data.reset_index().sort_values(by='long', ascending=False).set_index('date')
        data_sorted_short_all_dates = df_data.reset_index().sort_values(by='short', ascending=False).set_index('date')
        df_temp = df_data.copy()
        df_temp['long_short'] = df_temp.long - 0.4 * df_temp.short
        data_sorted_long_short_all_dates = df_temp.reset_index().sort_values(by='long_short',
                                                                             ascending=False).set_index('date')

        # empty dataframe that will be used to store results
        df_permnos_to_buy = pd.DataFrame(columns=strategies, index=sorted(set(df_data.index)))

        def get_weights(data_sorted, list_permnos_to_buy):
            if equi_weighted:
                weight_permnos = np.ones(len(list_permnos_to_buy)) / len(list_permnos_to_buy)
            else:  # make weights proportional to the signal we consider
                weight_permnos = data_sorted.loc[list_permnos_to_buy].values.T[0]
                weight_permnos = weight_permnos / weight_permnos.sum()
            return weight_permnos

        # calculate proportional weights and stocks to buy in each bin
        def perform_bin_strategy(data_sorted):
            bin = int(strat.split('_')[0])  # format must be '4_bins_...' for the 4th bin
            n_stocks = len(data_sorted.index)
            list_permnos_to_buy = list(data_sorted.index)[
                                  round((bin - 1) * n_stocks / bins):round(bin * n_stocks / bins)]
            weight_permnos = get_weights(data_sorted, list_permnos_to_buy)
            return list_permnos_to_buy, weight_permnos

        for date in df_data.index:
            data_sorted_long = data_sorted_long_all_dates.loc[date].set_index('PERMNO')[['long']]
            data_sorted_short = data_sorted_short_all_dates.loc[date].set_index('PERMNO')[['short']]
            data_sorted_long_short = data_sorted_long_short_all_dates.loc[date].set_index('PERMNO')[['long_short']]

            for strat in strategies:
                if strat == '10_max_long':
                    list_permnos_to_buy = list(data_sorted_long.index[:10])
                    weight_permnos = get_weights(data_sorted_long, list_permnos_to_buy)
                elif strat == '20_max_long':
                    list_permnos_to_buy = list(data_sorted_long.index[:20])
                    weight_permnos = get_weights(data_sorted_long, list_permnos_to_buy)
                elif strat == '2_max_long':
                    list_permnos_to_buy = list(data_sorted_long.index[:2])
                    weight_permnos = get_weights(data_sorted_long, list_permnos_to_buy)
                elif strat == 'threshold':
                    list_permnos_to_buy = list(df_data[df_data.long >= 0.75].index)
                    weight_permnos = get_weights(data_sorted_long, list_permnos_to_buy)
                elif 'bins_long' in strat:
                    list_permnos_to_buy, weight_permnos = perform_bin_strategy(data_sorted_long)
                elif 'bins_short' in strat:
                    list_permnos_to_buy, weight_permnos = perform_bin_strategy(data_sorted_short)
                elif 'bins_long_short' in strat:
                    list_permnos_to_buy, weight_permnos = perform_bin_strategy(data_sorted_long_short)
                else:
                    raise NotImplementedError('The strategy {} is not implemented'.format(strat))
                df_permnos_to_buy.loc[date][strat] = list_permnos_to_buy, weight_permnos
        self.log('Signals created')
        return df_permnos_to_buy

    def _make_df_for_bckt(self, pred):
        """
        Instantiates self._df_all_data with a Dataframe containing prices and predictions and sorts it
        chronologically
                * Example:

                PERMNO     RET      long      hold     short
        date
        20181030  83387.0    1.023  0.149458  0.353090  0.497452
        20181030  84207.0  1.02  0.404111  0.385122  0.210767

        :param pred: numpy array shape (N_samples,3)
        :return: Nothing
        """
        self.log('Joining prices data and predictions')
        df_signals = pd.DataFrame(index=self._df_all_data.index, data=pred, columns=['long', 'hold', 'short'])
        self._df_all_data = pd.concat([self._df_all_data, df_signals], axis=1)
        self._df_all_data = self._df_all_data.sort_index()

    def restore_output_op(self, sess, latest=False):
        """
        Restores tensors : x and output and instanciates self._x and self._output
        :param sess: tf.Session
        :param latest: bool: set to True to retrive the latest checkpoint.meta file in the specified folder

        Instantiates self._output, self._x, self._phase_train, self._dropout  with tensors retrieved from checkpoint file
        """
        saver = tf.train.import_meta_graph(self._path_model_to_restore)
        folder = os.path.dirname(self._path_model_to_restore)
        if latest:
            file_to_restore = tf.train.latest_checkpoint(folder)
        else:
            file_to_restore = self._path_model_to_restore.replace('.meta', '')
        saver.restore(sess, file_to_restore)

        graph = tf.get_default_graph()

        self._output = graph.get_tensor_by_name('{}/output:0'.format(self._name_network))
        self._x = graph.get_tensor_by_name('{}/x:0'.format(self._name_network))
        self._phase_train = graph.get_tensor_by_name('{}/phase_train:0'.format(self._name_network))
        self._dropout = graph.get_tensor_by_name('{}/dropout:0'.format(self._name_network))

    def run_predictions(self, X):
        """
        Restores a model saved with tensorflow and predicts signals
        :param X: numpy array: shape (N_samples,42,42,5) (if we consider 42 pixels and 5 channels)
        :return: predictions as a numpy array shape (N_samples,3)
        """
        self.log('Restoring model and making the predictions')
        sess = tf.Session()
        graph = tf.get_default_graph()
        self.restore_output_op(sess)
        self.log('Model Restored, launching output operation')

        # we run predictions in batches to avoid killing the kernel
        size_1_image = np.prod(X[0].shape)
        limit_size = 20 * 42 * 42 * 5
        size_batch = int(limit_size / size_1_image) + 1
        pred = np.zeros((0, 3))
        self.log('To avoid killing the kernel, we run predictions in {} batches'.format(round(len(X) / size_batch)))
        for batch in range(0, len(X), size_batch):
            X_batch = X[batch:batch + size_batch]
            pred_batch = sess.run(self._output,
                                  feed_dict={self._x: X_batch, self._phase_train: False, self._dropout: 0.})
            pred = np.concatenate([pred, pred_batch])

        self.log('Predictions computed')
        sess.close()
        return pred

    def plot_backtest(self):
        self.df_strats.plot(grid=True,figsize=(16,12),title='Backtest for different strategies')

    def get_df_all_data(self):
        """
        - Unpickles data files and gets samples
        - Keeps only the dates we want

        - Instanciates self._df_all_data: Example
                                  PERMNO       RET      long      hold     short
                    date
                    20181101  79237.0  0.952770  0.019922  0.458449  0.521629
                    20181101  78877.0  0.991597  0.080810  0.658793  0.260397
        """
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
            self.log(
                'first_date: {}, last_date: {}'.format(df_all_data_one_batch.index[0], df_all_data_one_batch.index[-1]))
            # Keeping only the dates we want for the backtest
            df_all_data_one_batch = df_all_data_one_batch.loc[self._start_date:self._end_date]
            self.log('new first_date: {}, new last_date: {}'.format(df_all_data_one_batch.index[0],
                                                                    df_all_data_one_batch.index[-1]))

            # in each of the pickle files the data is sorted in chronologic order we resort just in case
            df_all_data_one_batch = df_all_data_one_batch.sort_index()
            X = np.concatenate([[sample for sample in df_all_data_one_batch['sample'].values]],
                               axis=0)  # need this to get an array
            # we only keep returns and permnos (which is the stock identifier)
            df_all_data_one_batch_for_bckt = df_all_data_one_batch[['PERMNO', 'RET']]
            if i == 0:
                X_res = X
                df_all_data = df_all_data_one_batch_for_bckt
            else:
                X_res = np.concatenate([X_res, X])
                df_all_data = pd.concat([df_all_data, df_all_data_one_batch_for_bckt])
        # now we must not change the order in X_res and df_all_data
        self._df_all_data = df_all_data
        return X_res

    @staticmethod
    def _format_df_strats(df_backt_results):
        """
        :param df_backt_results:pd.DataFrame: results of the backtest data by date
        :return: pd.DataFrame cumulative results with a datetime index and an extra column 'Cash' which is the SPX
        """
        df_backt_results = df_backt_results.shift(
            1).dropna()  # because returns in reality do not happen just when we buy, but on next week
        df_strats = df_backt_results.astype(np.float64)
        df_strats = integer_to_timestamp_date_index(df_strats)
        df_strats = df_strats.cumprod() / df_strats.iloc[0]

        # we get SPX data and add it to the dataframe
        df_spx = pd.read_csv('data/^GSPC.csv', index_col=0, usecols=['Date', 'Close'])
        df_spx.index.name = 'date'
        df_spx.columns = ['Cash']  # Columns that will be used to compute sharpe ratio
        df_spx.index = pd.to_datetime(df_spx.index)
        df_strats = pd.merge(df_strats, df_spx, how='left', on='date')
        df_strats['Cash'] = df_strats['Cash'] / df_strats['Cash'].iloc[0]
        return df_strats

    def log(self, msg):
        log(msg, environment=self.__LOGGER_ENV)
