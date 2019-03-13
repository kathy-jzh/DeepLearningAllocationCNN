from ipyLogger import get_logger
import sys
import os
import pickle
import ezhc as hc
import pandas as pd


def plot_highstock_with_table(df, title=''):
    # Todo add more args
    g = hc.Highstock()

    g.chart.width = 900
    g.chart.height = 600
    g.legend.enabled = True
    g.legend.layout = 'horizontal'
    g.legend.align = 'center'
    g.legend.maxHeight = 100
    g.tooltip.enabled = True
    g.tooltip.valueDecimals = 2
    g.exporting.enabled = True

    g.chart.zoomType = 'xy'
    g.title.text = title
    g.subtitle.text = 'Subtitle ? '

    g.plotOptions.series.compare = 'percent'

    g.xAxis.gridLineWidth = 1.0
    g.xAxis.gridLineDashStyle = 'Dot'
    g.yAxis.gridLineWidth = 1.0
    g.yAxis.gridLineDashStyle = 'Dot'

    g.series = hc.build.series(df)

    return g.plot_with_table_1(save=False, dated=True, version='latest')


def integer_to_timestamp_date_index(df):
    df.index.name = 'date'
    df = df.reset_index()
    df.date = pd.to_datetime(df.date.apply(lambda x: '{}-{}-{}'.format(str(x)[:4], str(x)[4:6], str(x)[6:])))
    df = df.set_index('date')
    return df


def log(msg, environment='', loglevel='info', **kwargs):
    """"""
    if __is_in_ipython():
        # '%(name)s | %(asctime)s | %(levelname)s:  %(message)s'
        formatter = kwargs.get('formatter', '%(asctime)s | %(levelname)s:  %(message)s')
        loglevel_limit = kwargs.get('loglevel_limit', None)
        logger = get_logger(environment, loglevel=loglevel_limit, formatter=formatter)
        if loglevel.lower() == 'info':
            logger.info(msg)
        elif loglevel.lower() == 'warning':
            logger.warning(msg)
        elif loglevel.lower() == 'critical':
            logger.critical(msg)
        elif loglevel.lower() == 'error':
            logger.error(msg)
        elif loglevel.lower() == 'debug':
            logger.debug(msg)
    else:
        print('{}.{}:  '.format(environment, loglevel.upper()), msg)


def dump_pickle(obj, path: str, protocol=pickle.HIGHEST_PROTOCOL, logger_env=None):
    """
    :param obj: obj to pickle
    :param path: path for the dump MUST BE with no suffix this function will automatically add '.pickle' in the name
    :param protocol: protocol for the pickling
    :param logger_env: for the logging
    """
    _ensure_or_make_dir(file_path=path, logger_env=logger_env)
    f = open(path + '.pickle', 'wb')
    try:
        pickle.dump(obj, f, protocol=protocol)
    except Exception as e:
        raise e
    finally:
        f.close()


def load_pickle(path: str, logger_env='Pickles'):
    """
    Unpickles an objects, returns it and logs before and after the loading
    :param path: str: path for the file to load
    :param logger_env: environment for ipyLogger: str
    :return: object of the pickle
    """
    # ensuring the file ends in '.pickle'
    path = path.replace('.pickle', '') + '.pickle'

    log('Unpickling data at {}'.format(path), environment=logger_env)
    f = open(path, 'rb')
    res = pickle.load(f)
    f.close()
    log('Data unpickled, type: {}'.format(res.__class__), environment=logger_env)
    return res


def remove_all_files_from_dir(directory, logger_env=None):
    _ensure_or_make_dir(directory=directory, logger_env=logger_env)
    files_to_remove = os.listdir(directory)
    log('Removing existing files in the folder {}'.format(directory), environment=logger_env)
    for f in files_to_remove:
        os.remove(os.path.join(directory, f))
    log('Files {} removed from folder {}'.format(files_to_remove, directory), environment=logger_env)


def _ensure_or_make_dir(directory=None, file_path=None, logger_env=None):
    assert (directory is not None) ^ (
            file_path is not None), 'only one of the args: file_path or directory, must be none to test or make a directory'
    directory = os.path.dirname(file_path) if file_path else directory
    if not os.path.exists(directory):
        log('Creating directory {}'.format(directory), environment=logger_env)
        os.makedirs(directory)


def __get_ipython_func():
    return sys.modules['__main__'].__dict__.get('get_ipython', None)


def __is_in_ipython():
    get_ipython = __get_ipython_func()
    if get_ipython is None:
        return False
    else:
        return get_ipython() is not None
