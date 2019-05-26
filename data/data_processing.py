import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from utils import load_pickle, log, dump_pickle, remove_all_files_from_dir
from config.config import DEFAULT_FILES_NAMES, DEFAULT_END_DATE, DEFAULT_START_DATE


class DataHandler:
    """
    DataHandler aims to:
        - Constructs features that will be used later
        - Builds labels using returns thresholds and volatility
        - Build long / hold / short strategies
        - Transform stock data with time series into images
        - Pickles dataframes on disk with keys/values
    """

    def __init__(self, encoding_method='GADF',
                 window_len=42,
                 image_size=42,
                 retrain_freq=5,
                 threshold_ret = (-0.014,0.014),
                 start_date: int = DEFAULT_START_DATE,
                 end_date: int = DEFAULT_END_DATE,
                 frac_of_stocks=1.,
                 minimum_volume=1e6,
                 stock_data_dir_path: str = 'data/2019_2010_stock_data',
                 dir_for_samples='data/cnn_samples/regular',
                 nb_of_stocks_by_file=50,
                 nb_files_to_read: int = 34,
                 ):
        """
        :param encoding_method: including GADF, GASF, MTF
        :param window_len: length of moving window
        :param image_size: size of image
        :param retrain_freq: frequency of retrain
        :param start_date: start date to consider for the stock data
        :param end_date: last date to consider for the stock data
        :param frac_of_stocks: fraction of data to utlize which ranges from start_date to end_date
        :param minimum_volume: minimum average daily volume for small stock filtering
        :param stock_data_dir_path: path to fetch stock data
        :param dir_for_samples: path to store transformed images
        :param nb_of_stocks_by_file: number of files to dump
        :param nb_files_to_read: number of files to read
        """
        self._window_len = window_len
        self._image_size = image_size
        self._retrain_freq = retrain_freq
        self._threshold_ret = threshold_ret
        self._encoding_method = encoding_method

        self._features = ['date', 'RET', 'ASKHI', 'BIDLO', 'VOL', 'sprtrn']
        self._min_volume = minimum_volume

        self._start_date = start_date
        self._end_date = end_date
        self._frac_of_stocks_to_get = frac_of_stocks
        self._nb_of_stocks_by_file = nb_of_stocks_by_file

        self._directory_for_samples = dir_for_samples
        self._stock_data_dir_path = stock_data_dir_path

        self._N_FILES_CRSP = nb_files_to_read
        self._LOGGER_ENV = 'image_encoding'

        self.df_data = None
        self._df_raw_data = None
        self._stocks_list = None

    def get_df_data(self):
        """
        - Identifies how many files it must read  (randomly) according to self._frac_of_stocks_to_get.
            Reads all the files if self._frac_of_stocks_to_get is 1. Use a value <1 for testing purposes
        - Filters data on Volume/dates
        - Constructs features that will be used later in our model

        :instanciates:
            * self.df_data: dataframe of all data with dates as index
            * self._stocks_list: all the unique permnos (stock identifiers) present in the data
            * self._df_raw_data: dataframe of the data as extracted from the database

        :return: Nothing
        """

        nb_files_to_get = max(round(self._frac_of_stocks_to_get * self._N_FILES_CRSP), 1)
        choices = np.random.choice(np.arange(1, self._N_FILES_CRSP + 1), nb_files_to_get, replace=False)
        file_names = ['stockdata_{}'.format(i) for i in choices]
        df_data = self._load_stock_data(file_names, data_dir_path=self._stock_data_dir_path,
                                        logger_env=self._LOGGER_ENV)
        self._df_raw_data = df_data
        df_data = self._filter_data(df_data)
        df_data = self.__rectify_prices(df_data)

        df_data = self._extract_features(df_data)
        self._stocks_list = np.unique(df_data.index)
        self.log('Data finalized in handler.df_data, number of stocks {}'.format(len(self._stocks_list)))
        self.df_data = self._get_data_between(df_data, self._start_date, self._end_date, self._LOGGER_ENV)


    def build_and_dump_images_and_targets(self):
        """
             * Builds images with the time series
             * Builds labels using returns thresholds and volatility
             * Pickles dataframes on disk with keys/values :
                - date as index
                - PERMNO: stock identifier
                - RET: returns that will be used for the backtest
                - samples: used for training and backtesting
                - close: one-hot encoded elements, Example [1,0,0] stands for the class 'long'
        """
        nb_stocks = len(self._stocks_list)
        n_files_to_dump = nb_stocks // self._nb_of_stocks_by_file + ((nb_stocks % self._nb_of_stocks_by_file) != 0)
        df_data_multi_index = self.df_data.reset_index(drop=False).set_index(['PERMNO', 'date'])
        self.log('***** Dumping data in {} different files'.format(n_files_to_dump))

        # Removing existing files in the folder
        remove_all_files_from_dir(self._directory_for_samples, logger_env=self._LOGGER_ENV)

        for batch in range(n_files_to_dump):
            btch_name = 'image_data_{}'.format(batch + 1)
            btch_stocks = self._stocks_list[batch * self._nb_of_stocks_by_file:(batch + 1) * self._nb_of_stocks_by_file]
            df_batch_data = self._extract_data_for_stocks(df_data_multi_index, btch_stocks)
            # Build Images and targets
            del df_res
            df_res = self._build_images_one_batch(df_batch_data, btch_name)
            # Sort by dates
            df_res = df_res.set_index('date').sort_index()
            # Dumping the pickle dataframe
            dump_pickle(df_res, os.path.join(self._directory_for_samples, btch_name), logger_env=self._LOGGER_ENV)


    @staticmethod
    def _build_close_returns(df, window_len=64, retrain_freq=5, up_return=0.0125, down_return=-0.0125,
                             buy_on_last_date=True):
        """
        :param up_return: threshold for long strategy
        :param down_return: threshold for short strategy
        :param buy_on_last_date: whether to buy on last date
        :return: strategy target list, backtesting dataframe, price return list, date list
        """
        n_sample = len(df)
        targets, prc_list, dates_list = [], [], []

        # Strategy of long / hold / short
        # Hold stands for a state that model can't make decision between long and short
        _long, _hold, _short = [1, 0, 0], [0, 1, 0], [0, 0, 1]

        rebalance_indexes = []
        df_rolling_ret = np.exp(np.log(df.RET).rolling(window=retrain_freq).sum())  # product of returns
        # print(df_rolling_ret)
        df_rolling_std = df.RET.rolling(window=window_len).std() * np.sqrt(252.)

        for i in range(window_len, n_sample, retrain_freq):
            j = i - 1 if buy_on_last_date else i

            price_return = df_rolling_ret.iloc[np.min([n_sample - 1, i - 1 + retrain_freq])]
            dates_list.append(df.index[j])
            prc_list.append(price_return)

            vol = df_rolling_std.iloc[j]

            if price_return - 1. > up_return * vol * 4:
                targets.append(_long)
            elif price_return - 1. < down_return * vol * 4:
                targets.append(_short)
            else:
                targets.append(_hold)

            # we keep the indexes of the dates when there will be a rebalance in the portfolio
            rebalance_indexes.append(j)
        df_for_backtest = df_rolling_ret.iloc[rebalance_indexes]

        return np.asarray(targets), df_for_backtest, prc_list, dates_list


    @staticmethod
    def _build_images_one_stock(df_one_permno, window_len, retrain_freq, encoding_method, image_size):
        """
        Encodes images as timeseries for one stock
        :param df_one_permno: dataframe of the timeseries of all data for one particular stock
        :param window_len: number of observations to consider (42 for 2 months)
        :param retrain_freq: lag to consider between making two samples
        :param encoding_method: method to encode the images
        :param image_size: final size of the image (using window_len*window_len will avoid any averaging)
        :return: np.ndarray of the samples of shape (N,window_len,window_len,M) where:
                - M is the number of features
                - N is the number of final samples ~ len(df_one_permno)/retrain_freq
        """

        n_days = df_one_permno.T.shape[-1]
        samples_list, dates_list, prc_list = [], [], []
        for i in range(window_len, n_days, retrain_freq):
            window_data = df_one_permno.T.iloc[:, i - window_len:i]

            # Use GADF algorithm to transform data
            if encoding_method == 'GADF':
                try:
                    from pyts.image import GADF
                    gadf = GADF(image_size)
                except:
                    from pyts.image import GramianAngularField
                    gadf = GramianAngularField(image_size, method='difference')
                samples_list.append(gadf.fit_transform(window_data).T)

            # Use GASF algorithm to transform data
            elif encoding_method == 'GASF':
                try:
                    from pyts.image import GASF
                    gasf = GASF(image_size)
                except:
                    from pyts.image import GramianAngularField
                    gasf = GramianAngularField(image_size, method='summation')
                samples_list.append(gasf.fit_transform(window_data).T)

            # Use MTF algorithm to transform data
            elif encoding_method == 'MTF':
                try:
                    from pyts.image import MTF
                    mtf = MTF(image_size)
                except:
                    from pyts.image import MarkovTransitionField
                    mtf = MarkovTransitionField(image_size)
                samples_list.append(mtf.fit_transform(window_data).T)
            else:
                raise BaseException('Method must be either GADF, GASF or MTF not {}'.format(encoding_method))
        samples_list = np.asarray(samples_list)
        return samples_list


    def _build_images_one_batch(self, df_batch_data, batch_name):
        """
        :param df_batch_data: dataframe of the timeseries of all data for a batch of stocks
        :param batch_name: name of the batch
        :return: pd.DataFrame with columns ['sample', 'date', 'RET', 'close']
        """
        self.log('Building Targets and Images for batch {}'.format(batch_name), )

        df_batch_data = df_batch_data.reset_index(drop=False).set_index(['PERMNO', 'date'])
        all_permnos = df_batch_data.index.levels[0]
        # The empty dataframe initialized
        columns_df_res = ['sample', 'date', 'RET', 'close']
        df_res = pd.DataFrame(columns=columns_df_res)

        for permno in all_permnos:
            df_one_permno = df_batch_data.loc[permno]
            samples_list = self._build_images_one_stock(df_one_permno, self._window_len,
                                                        self._retrain_freq, self._encoding_method,
                                                        self._image_size)

            labels_array, df_for_backtest, prc_list, dates_list = self._build_close_returns(df_one_permno,
                                                                                            self._window_len,
                                                                                            self._retrain_freq,up_return=self._threshold_ret[1],
                                                                                            down_return=self._threshold_ret[0])

            # building dataframe
            df_res_one_permno = pd.DataFrame(columns=columns_df_res)

            for k, date in enumerate(dates_list):
                data = [samples_list[k], date, prc_list[k], labels_array[k]]
                row_df = pd.DataFrame(columns=columns_df_res, data=[data])
                df_res_one_permno = pd.concat([df_res_one_permno, row_df],sort=True)
            df_res_one_permno['PERMNO'] = permno
            df_res = pd.concat([df_res, df_res_one_permno])

        self.log('Targets and Images for batch {} are built'.format(batch_name), )

        return df_res

    def _filter_data(self, df_data):
        """
        Only keeps stocks that meet specified criteria:
            - Minimum average daily volume of  self._min_volume
            - List of stocks to keep or to avoid
        :param df_data:  pd.dataframe
         - Columns: ['date', 'TICKER', 'COMNAM', 'BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET','SHROUT', 'sprtrn']
         - Index PERMNO
        :return: filtered dataframe
        """
        df_filter = df_data.reset_index()[['PERMNO', 'VOL']]
        df_filter = df_filter.groupby('PERMNO').mean().sort_values('VOL', ascending=False)
        df_filter = df_filter[df_filter.VOL >= self._min_volume]
        list_stocks = df_filter.index
        # if we need to choose only some stocks
        df_data = df_data.loc[list_stocks]
        return df_data

    def __rectify_prices(self, df_data):
        """
        In our database prices are often given negative as a symbol when the MID is not really accurate
        we need to rectify prices to positive values
        :param df_data:  pd.dataframe
         - Columns: ['date', 'TICKER', 'COMNAM', 'BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET','SHROUT', 'sprtrn']
         - Index PERMNO
        :return:
        """
        self.log(
            'Rectifying {} negative prices out of {} prices'.format(len(df_data[df_data.PRC <= 0]), len(df_data.PRC)))
        df_data.PRC = np.abs(df_data.PRC)
        return df_data

    def _extract_features(self, df_data: pd.DataFrame):
        """
        Computes the features we want and keep only what is necessary for the neural network
        Features:
            - Volume is adjusted in order to keep consistent value in case of stock splits
            - Bid and ask prices are transformed as followed : BIDLO ->(PRC - BIDLO)/PRC and (ASKHI - PRC) / PRC

        :param df_data: Example:
                        date	TICKER	COMNAM	            BIDLO	ASKHI	        PRC	        VOL	        RET 	SHROUT	    sprtrn
            PERMNO
            36468	20100104	SHW	SHERWIN WILLIAMS CO	61.17000	62.14000	61.67000	1337900.0	0.000324	113341.0	0.016043
            36468	20100105	SHW	SHERWIN WILLIAMS CO	59.55000	61.86000	60.21000	3081500.0	-0.023674	113341.0	0.003116

        :return: dataframe with modified features and only self._features as columns
        """
        df_data.BIDLO = (df_data.BIDLO - df_data.PRC) / df_data.PRC
        df_data.ASKHI = (df_data.ASKHI - df_data.PRC) / df_data.PRC
        df_data.VOL = df_data.VOL / df_data.SHROUT  # to compensate for stock splits
        df_data.RET = df_data.RET + 1.

        columns_to_get = self._features
        return df_data[columns_to_get]

    @staticmethod
    def _get_data_between(df: pd.DataFrame, start_date: int = DEFAULT_START_DATE, end_date: int = DEFAULT_END_DATE,
                          logger_env='image_encoding'):
        """
        :param df: dataframe with PERMNO as index, date must be a column in the dataframe
        :param start_date: first boundary example 20150101
        :param end_date: second boundary
        :return: dataframe with new dates
        """
        log('Getting data between {} and {}'.format(start_date, end_date), environment=logger_env)
        assert 'date' in df.columns, 'date must be one of the columns but columns are {}'.format(df.columns)

        df_res = df[df.date >= start_date]
        df_res = df_res[df_res.date <= end_date]

        sorted_dates = np.sort(df_res.date.values)
        new_start_date, new_end_date = sorted_dates[0], sorted_dates[-1]
        log('New boundary dates: {} and {}'.format(new_start_date, new_end_date), environment=logger_env)

        return df_res

    @staticmethod
    def _load_stock_data(file_names: list = DEFAULT_FILES_NAMES, data_dir_path: str = 'data',
                         logger_env: str = 'Pickling'):
        """
        :return: dataframe with all data: Example:

                PERMNO(which is index)     date TICKER         COMNAM      DIVAMT  NSDINX   BIDLO  \
                91707                  20100104    MVO        M V OIL TRUST     NaN     NaN  20.515
                91707                  20100105    MVO        M V OIL TRUST     NaN     NaN  21.140


                ASKHI      PRC       VOL    BID    ASK    sprtrn
                21.1950  21.1501   86300.0  21.15  21.19  0.016043
                21.5700  21.5700   70300.0  21.53  21.71  0.003116
        """
        assert len(file_names) >= 1, 'the list of file names is <1'
        for i, file_name in enumerate(file_names):
            if i == 0:
                df_res = load_pickle(os.path.join(data_dir_path, file_name), logger_env=logger_env)
                assert isinstance(df_res, pd.DataFrame), 'the data from {} is not a DataFrame but {}'.format(file_name,
                                                                                                             df_res.__class__)
            else:
                df_temp = load_pickle(os.path.join(data_dir_path, file_name), logger_env=logger_env)
                assert isinstance(df_temp, pd.DataFrame), 'the data from {} is not a DataFrame but {}'.format(file_name,
                                                                                                              df_temp.__class__)
                df_res = pd.concat([df_res, df_temp])

        df_res = df_res.set_index('PERMNO')
        return df_res

    def log(self, msg):
        log(msg, environment=self._LOGGER_ENV)

    @staticmethod
    def _extract_data_for_stocks(df_data_multi_ind: pd.DataFrame, list_stocks: list):
        """
        :param df_data_multi_ind: Multu-Index Dataframe

                * Example:
                                            PRC	ASKHI	BIDLO	VOL
                    PERMNO	date
                    92279	20150102	18.4100	18.6400	18.3241	39626.0
                            20150105	18.1300	18.2700	18.0800	153953.0
                            20150106	18.0100	18.1700	17.8900	1356976.0

        :param list_stocks: list of stocks to keep
        :return: Example:
                          PERMNO       RET     ASKHI     BIDLO       VOL    sprtrn
                date
                20150102   38703  0.997811  0.008958 -0.009260  2.255533 -0.000340
                20150105   38703  0.972578  0.021805 -0.001880  2.891599 -0.018278
        """
        df_res = df_data_multi_ind.loc[list_stocks]
        df_res = df_res.reset_index(level=0, drop=False)
        return df_res


    def show_multichannels_images(self):
        """
        Plots a multi dimensional timeseries encoded as n_dim images for one stock
        and the time window specified for the handler object
        """
        assert len(self._stocks_list) > 0, 'There is less than one permno available'
        permno = self._stocks_list[0]
        df_window_data = self.df_data.loc[permno][-self._image_size:]
        dates = df_window_data.date.iloc[0], df_window_data.date.iloc[-1]

        msg = '### Showing encoded images with method {} for the stock {} between {} and {} '.format(
            self._encoding_method, permno, dates[0], dates[1])
        display(Markdown(msg))
        self._show_images(df_window_data)


    def _show_images(self, df_window_data):
        """
        Plots a multi dimensional timeseries encoded as an image
        :param df_window_data: timeseries we want to encode as an image
        """
        data = df_window_data.reset_index().set_index('date').drop('PERMNO', axis=1).T
        channels = list(data.index)
        if self._encoding_method == 'GADF':
            try:
                from pyts.image import GADF
                gadf = GADF(self._image_size)
            except:
                from pyts.image import GramianAngularField
                gadf = GramianAngularField(self._image_size, method='difference')
            image_data = (gadf.fit_transform(data).T)

        elif self._encoding_method == 'GASF':
            try:
                from pyts.image import GASF
                gasf = GASF(self._image_size)
            except:
                from pyts.image import GramianAngularField
                gasf = GramianAngularField(self._image_size, method='summation')
            image_data = (gasf.fit_transform(data).T)
        elif self._encoding_method == 'MTF':
            try:
                from pyts.image import MTF
                mtf = MTF(self._image_size)
            except:
                from pyts.image import MarkovTransitionField
                mtf = MarkovTransitionField(self._image_size)
            image_data = (mtf.fit_transform(data).T)
        else:
            raise BaseException('Method must be either GADF, GASF or MTF not {}'.format(self._encoding_method))

        num_channels = image_data.shape[-1]
        plt.figure(figsize=(12, 14))
        for j in range(1, num_channels + 1):
            channel = image_data[:, :, j - 1]
            plt.subplot(int(num_channels / 2) + 1, 2, j)
            plt.imshow(channel, cmap='rainbow', origin='lower')
            plt.xlabel('$time$')
            plt.ylabel('$time$')
            plt.title(channels[j - 1])
            plt.tight_layout()

        plt.show()


def get_training_data_from_path(samples_path='data/cnn_samples/regular',
                                targets_type='close',
                                train_val_size=2 / 3.,
                                train_size=0.75,
                                logger_env='Training',
                                ):
    """
    Unpickles data and formats it for training

    :param samples_path: path for the folder with the data we need: only the files we need need to be in this folder
    :param targets_type: str: the targets to consider for this training
    :param train_val_size: training/(training+validation)
    :param train_size: size of (training+validation)/(training+test+validation)
    :return: X_train, X_val, X_test, Y_train, Y_val, Y_test as numpy arrays
    """
    # list of files in the folder: samples_paths
    list_file_names = os.listdir(samples_path)
    assert len(list_file_names) >= 1, 'The number of files in the folder {} is probably 0, it must be >=1'.format(
        samples_path)
    log('******* Getting data from folder: {}, Nb of files : {}, First file : {}'.format(samples_path,
                                                                                         len(list_file_names),
                                                                                         list_file_names[0]),
        logger_env)

    for i, file_name in enumerate(list_file_names):
        path = os.path.join(samples_path, file_name)
        df_all_data = load_pickle(path, logger_env=logger_env)
        df_all_data = df_all_data.sort_index()  # In case it is not sorted

        log('first_date: {}, last_date: {}'.format(df_all_data.index[0], df_all_data.index[-1]),
            environment=logger_env)
        # in each of the pickle files the data is sorted in chronologic order
        X = np.concatenate([[sample for sample in df_all_data['sample'].values]], axis=0)  # need this to get an array
        Y = np.concatenate([[sample for sample in df_all_data[targets_type].values]], axis=0)
        n_samples = Y.shape[0]

        stop_1 = round(train_val_size * n_samples * train_size)
        stop_2 = round(n_samples * train_size)
        if i == 0:
            X_train = X[:stop_1]
            Y_train = Y[:stop_1]

            X_val = X[stop_1:stop_2]
            Y_val = Y[stop_1:stop_2]

            X_test = X[stop_2:]
            Y_test = Y[stop_2:]
        else:

            X_train = np.concatenate([X_train, X[:stop_1]])
            Y_train = np.concatenate([Y_train, Y[:stop_1]])

            X_val = np.concatenate([X_val, X[stop_1:stop_2]])
            Y_val = np.concatenate([Y_val, Y[stop_1:stop_2]])

            X_test = np.concatenate([X_test, X[stop_2:]])
            Y_test = np.concatenate([Y_test, Y[stop_2:]])
        # if predict:

    log('Nb of Samples in training:{}, validation: {}, test:{}'.format(len(Y_train), len(Y_val), len(Y_test)),
        environment=logger_env)
    # Distributions of Labels
    train_d = np.sum(Y_train, axis=0) / len(Y_train)
    val_d = np.sum(Y_val, axis=0) / len(Y_val)
    tst_d = np.sum(Y_test, axis=0) / len(Y_test)
    text_template = 'long: {:5.2f}%, hold: {:5.2f}%, short: {:5.2f}%'
    log('Training Distribution of Labels :' + text_template.format(train_d[0] * 100, train_d[1] * 100,
                                                                   train_d[2] * 100),
        environment=logger_env)
    log('Validation Distribution of Labels :' + text_template.format(val_d[0] * 100, val_d[1] * 100, val_d[2] * 100),
        environment=logger_env)
    log('Test Distribution of Labels :' + text_template.format(tst_d[0] * 100, tst_d[1] * 100, tst_d[2] * 100),
        environment=logger_env)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


