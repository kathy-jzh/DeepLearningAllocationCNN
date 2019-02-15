# TODO:  tranform time serie in images here
# todo build functions in utils to unppickle and pickle data easily

import numpy as np
import pandas as pd
import os
from pyts.image import GADF, GASF, MTF
from pykalman import KalmanFilter
import time


from utils import load_pickle, log, dump_pickle
from config.hyperparams import DEFAULT_FILES_NAMES, DEFAULT_END_DATE, DEFAULT_START_DATE



class DataHandler:  # todo add multi processing
    def __init__(self, encoding_method='GADF', window_len=64, image_size=16, retrain_freq=5,
                 start_date: int = DEFAULT_START_DATE,
                 end_date: int = DEFAULT_END_DATE, frac_of_stocks=1., stock_data_dir_path: str = 'data',
                 dir_for_samples='data/cnn_samples/regular', nb_of_stocks_by_file=50
                 ):

        self._window_len = window_len
        self._image_size = image_size
        self._retrain_freq = retrain_freq
        # self.samples_dict = {} # todo see if this is not too big
        self._n_samples = 0
        self._encoding_method = encoding_method
        self._targets_methods = ['VWAP']

        self._start_date = start_date
        self._end_date = end_date
        self._frac_of_stocks_to_get = frac_of_stocks
        self._nb_of_stocks_by_file = nb_of_stocks_by_file

        self._directory_for_samples = dir_for_samples
        self._stock_data_dir_path = stock_data_dir_path

        self._logger_env = 'image_encoding'

        self.df_data = None
        self._stocks_list = None
        self._N_FILES_CRSP = 27

    def get_df_data(self):
        """
        identifies how many files it must read (randomly)
        Reads all the files if self._frac_of_stocks_to_get is 1.

        :instanciates:
            * self.df_data: dataframe of all data with dates as index
            * sekf._stocks_list: all the uniqu epermno present in the data

        :return: Nothing
        """
        nb_files_to_get = max(round(self._frac_of_stocks_to_get * self._N_FILES_CRSP),
                              1)  # todo modify later when we will have a different dataset
        choices = np.random.choice(np.arange(1, self._N_FILES_CRSP + 1), nb_files_to_get, replace=False)
        file_names = ['stockData_{}'.format(i) for i in choices]
        df_data = self._load_stock_data(file_names, data_dir_path=self._stock_data_dir_path,
                                        logger_env=self._logger_env)
        df_data = self._get_data_from_stocks(df_data, ['10026'])  # todo

        self._stocks_list = np.unique(df_data.index)
        log('Data finalized in attribute df_data, number of stocks {}'.format(len(self._stocks_list)),
            environment=self._logger_env)
        self.df_data = self._get_data_between(df_data, self._start_date, self._end_date, self._logger_env)

    # todo dump images via the object self.samples ? at once ?
    def build_and_dump_images_and_targets(self, use_smoothed_data=False):
        """
             * Selects only the data we want (dates and stocks)
             * Builds images with the timeseries
             * Builds targets with the specified methods
             * Pickles dictionnaries on disk with keys/values :
                - batch_name: str
                - n_samples: int
                - samples: (numpy array)
                - VWAP_targets (for VWAP if it is one of the specified methods)
                - ..._targets if ther is other targets
                - df_original_data

        Builds image data
        :param use_smoothed_data: id we use KalmanFilter # todo think about FFT and wavelet

        """
        nb_stocks = len(self._stocks_list)
        n_files_to_dump = nb_stocks // self._nb_of_stocks_by_file + ((nb_stocks % self._nb_of_stocks_by_file) != 0)
        df_data_multi_index = self.df_data.reset_index(drop=False).set_index(['PERMNO', 'date'])
        log('***** Dumping data in {} different files'.format(n_files_to_dump), environment=self._logger_env)

        # Removing existing files in the folder
        files_to_remove = os.listdir(self._directory_for_samples)
        log('Removing existing files in the folder {}'.format(self._directory_for_samples),environment=self._logger_env)
        for f in files_to_remove:
            os.remove(os.path.join(self._directory_for_samples,f))
        log('Files {} removed from folder {}'.format(files_to_remove,self._directory_for_samples),environment=self._logger_env)


        for batch in range(n_files_to_dump):
            batch_name = 'image_data_{}'.format(batch + 1)
            batch_stocks = self._stocks_list[
                           batch * self._nb_of_stocks_by_file:(batch + 1) * self._nb_of_stocks_by_file]
            df_batch_data = self._extract_data_for_stocks(df_data_multi_index, batch_stocks)
            # build images
            dict_to_pickle = self._build_images_one_batch(df_batch_data, batch_name,
                                                          use_smoothed_data=use_smoothed_data)
            # build targets
            dict_targets = self._get_targets_one_batch(df_batch_data, batch_name)
            dict_to_pickle.update({**dict_targets,'df_original_data':df_batch_data})

            dump_pickle(dict_to_pickle, os.path.join(self._directory_for_samples, batch_name))

        self.__delete_df_data_from_memory()

    def _get_targets_one_batch(self, df_batch_data: pd.DataFrame, batch_name: str):
        dict_targets = {}
        log('***** Building Targets for batch {}, methods will be {}'.format(batch_name, self._targets_methods),
            environment=self._logger_env)
        assert len(self._targets_methods) >= 1, 'The number of methods specified must be >=1'
        for method in self._targets_methods:
            log('Target building method {}'.format(method), environment=self._logger_env)
            if method == 'VWAP':
                # TODO put the up and down returns as params
                labels_array = self._build_VWAP_returns(df_batch_data, self._window_len, self._retrain_freq,
                                                        up_return=0.0125, down_return=-0.0125)
                dict_targets.update({'VWAP_targets': labels_array})
            elif method == 'close':
                labels_array = None  # todo
                log('method close is not yet implemented!!!!!', environment=self._logger_env, loglevel='error')
            else:
                raise BaseException('So far the targets can only be computed by close prices or VWAP')
        log('*****Targets for batch {} are built'.format(batch_name), environment=self._logger_env)
        return dict_targets

    @staticmethod
    def _build_VWAP_returns(df, window_len=64, retrain_freq=5, up_return=0.0125, down_return=-0.0125):
        data = df[['PRC', 'VOL']]
        n_sample = len(data)
        targets = []

        _long = [1, 0, 0]
        _hold = [0, 1, 0]
        _short = [0, 0, 1]

        for i in range(window_len, n_sample, retrain_freq):
            if data.VOL.iloc[i - retrain_freq:i].values.sum() > 0:
                lastVWAP = np.average(data.PRC.iloc[i - retrain_freq:i].values,
                                      weights=data.VOL.iloc[i - retrain_freq:i].values)
            else:
                lastVWAP = data.PRC.iloc[i]

            if data.VOL.iloc[i:np.min([n_sample - 1, i + retrain_freq])].values.sum() > 0:
                nextVWAP = np.average(data.PRC.iloc[i:np.min([n_sample - 1, i + retrain_freq])].values,
                                      weights=data.VOL.iloc[i:np.min([n_sample - 1, i + retrain_freq])].values)
            else:
                nextVWAP = data.PRC.iloc[np.min([n_sample - 1, i + retrain_freq])]

            VWAPReturn = (nextVWAP - lastVWAP) / lastVWAP

            if VWAPReturn > up_return:
                targets.append(_long)
            elif VWAPReturn < down_return:
                targets.append(_short)
            else:
                targets.append(_hold)

        return np.asarray(targets)

    def _build_images_one_batch(self, df_batch_data, batch_name, use_smoothed_data=False):
        data = df_batch_data.drop('PERMNO', axis=1).T
        n_days = data.shape[-1]
        samples_list = []
        log('****** Building images with method {},Batch: {} Data Shape: {}'.format(self._encoding_method, batch_name,
                                                                                    data.shape),
            environment=self._logger_env)
        for i in range(self._window_len, n_days, self._retrain_freq):
            window_data = data.iloc[:, i - self._window_len:i]

            # todo understand how this works
            if use_smoothed_data:
                Smoother = KalmanFilter(n_dim_obs=window_data.shape[0], n_dim_state=window_data.shape[0],
                                        em_vars=['transition_matrices', 'observation_matrices',
                                                 'transition_offsets', 'observation_offsets',
                                                 'transition_covariance', 'observation_convariance',
                                                 'initial_state_mean', 'initial_state_covariance'])
                measurements = window_data.T.values
                Smoother.em(measurements, n_iter=5)
                window_data, _ = Smoother.smooth(measurements)
                window_data = window_data.T

            if self._encoding_method == 'GADF':
                gadf = GADF(self._image_size)
                samples_list.append(gadf.fit_transform(window_data).T)

            elif self._encoding_method == 'GASF':
                gasf = GASF(self._image_size)
                samples_list.append(gasf.fit_transform(window_data).T)
        n_samples = len(samples_list)
        log('Images in batch {} are built, n_samples {}'.format(batch_name, n_samples),
            environment=self._logger_env)
        # self.samples_dict.update({'batch_name':batch_name,'n_samples':n_samples,'samples': np.asarray(samples_list)})
        return {'batch_name': batch_name, 'n_samples': n_samples, 'samples': np.asarray(samples_list)}

    @staticmethod
    def _get_data_from_stocks(df: pd.DataFrame, stocks_list: list):
        # TODO
        return df

    def __delete_df_data_from_memory(self):
        self.df_data = None

    @staticmethod
    def _get_data_between(df: pd.DataFrame, start_date: int = DEFAULT_START_DATE, end_date: int = DEFAULT_END_DATE,
                          logger_env='image_encoding'):
        """

        :param df: dataframe with PERMNO as index, date must be a column in the dataframe
        :param start_date: int Example:
        :param end_date:
        :return:
        """
        log('Getting data between {} and {}'.format(start_date, end_date), environment=logger_env)
        assert 'date' in df.columns, 'date must be one of the columns but columns are {}'.format(df.columns)
        df_res = df[df.date >= start_date]
        df_res = df_res[df_res.date <= end_date]
        return df_res

    @staticmethod
    def _load_stock_data(file_names: list = DEFAULT_FILES_NAMES, data_dir_path: str = 'data',
                         logger_env: str = 'Pickling'):
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

        # df_res = df_res.reset_index(drop=False).set_index(['PERMNO', 'date'])

        return df_res

    @staticmethod
    def _extract_data_for_stocks(df_data_multi_ind: pd.DataFrame, list_stocks: list):
        df_res = df_data_multi_ind.loc[list_stocks]
        df_res = df_res.reset_index(level=0, drop=False)
        return df_res


def get_training_data_from_path(samples_path='data/cnn_samples/regular',
                                targets_type='VWAP_targets',
                                train_val_size=2 / 3.,
                                train_size=0.75,
                                logger_env='Training'
                                ):

    """
    :param samples_path: path for the folder with the data we need: only the files we need need to be in this folder
    :param targets_type: str: the targets to consider for this training
    :param train_val_size: training/(training+validation)
    :param train_size: size of (training+validation)/(training+test+validation)
    :return: X_train, X_val, X_test, Y_train, Y_val, Y_test as numpy arrays
    """
    # list of files in the folder: samples_paths
    list_file_names = os.listdir(samples_path)
    assert len(list_file_names)>=1, 'The number of files in the folder {} is probably 0, it must be >=1'.format(samples_path)
    log('******* Getting data from folder: {}, Nb of files : {}, First file : {}'.format(samples_path, len(list_file_names),
                                                                                 list_file_names[0]), logger_env)
    for i,file_name in enumerate(list_file_names):
        path = os.path.join(samples_path, file_name)
        dict_from_pickle = load_pickle(path, logger_env=logger_env)

        X = dict_from_pickle['samples']
        Y = dict_from_pickle[targets_type]
        n_samples = dict_from_pickle['n_samples']
        if i==0:
            X_train = X[:round(train_val_size * n_samples * train_size)]
            Y_train = Y[:round(train_val_size * n_samples * train_size)]

            X_val = X[round(train_val_size * n_samples * train_size):round(n_samples * train_size)]
            Y_val = Y[round(train_val_size * n_samples * train_size):round(n_samples * train_size)]

            X_test = X[:round(n_samples * train_size)]
            Y_test = Y[:round(n_samples * train_size)]
        else:

            X_train = np.concatenate([X_train, X[:round(train_val_size * n_samples * train_size)] ])
            Y_train = np.concatenate([Y_train,  Y[:round(train_val_size * n_samples * train_size)] ])

            X_val= np.concatenate([X_val, X[round(train_val_size * n_samples * train_size):round(n_samples * train_size)] ])
            Y_val = np.concatenate([Y_val, Y[round(train_val_size * n_samples * train_size):round(n_samples * train_size)] ])

            X_test = np.concatenate([X_test,  X[:round(n_samples * train_size)] ])
            Y_test = np.concatenate([Y_test,  Y[:round(n_samples * train_size)] ])

    train_d = np.sum(Y_train, axis=0) / len(Y_train)
    val_d = np.sum(Y_val, axis=0) / len(Y_val)
    tst_d = np.sum(Y_test, axis=0) / len(Y_test)

    text = 'long: {:1f}%, hold: {:1f}%, short: {:1f}%'
    log('Training Distribution of Labels :'+text.format(train_d[0]*100,train_d[1]*100,train_d[2]*100),environment=logger_env)
    log('Validation Distribution of Labels :'+text.format(val_d[0]*100,val_d[1]*100,val_d[2]*100),environment=logger_env)
    log('Test Distribution of Labels :'+text.format(tst_d[0]*100,tst_d[1]*100,tst_d[2]*100),environment=logger_env)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def generate_dummy_data(batch_size):
    size_third = int(batch_size / 3.)
    rest = batch_size - 3 * size_third

    data_x_label_1 = np.random.uniform(-1, 0.9, (size_third, 16, 16, 4))
    data_x_label_2 = np.random.uniform(-0.95, 0.95, (size_third, 16, 16, 4))
    data_x_label_3 = np.random.uniform(-0.9, 1, (size_third + rest, 16, 16, 4))

    data_x_label_1 = np.asarray(data_x_label_1, np.float32)
    data_x_label_2 = np.asarray(data_x_label_2, np.float32)
    data_x_label_3 = np.asarray(data_x_label_3, np.float32)

    data_y_label_1 = np.asarray([[1, 0, 0] for i in range(size_third)], np.float32)
    data_y_label_2 = np.asarray([[0, 1, 0] for i in range(size_third)], np.float32)
    data_y_label_3 = np.asarray([[0, 0, 1] for i in range(size_third + rest)], np.float32)

    data_x = np.concatenate((data_x_label_1, data_x_label_2, data_x_label_3))
    data_y = np.concatenate((data_y_label_1, data_y_label_2, data_y_label_3))

    return data_x, data_y
