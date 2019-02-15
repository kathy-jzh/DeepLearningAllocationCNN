from ipyLogger import get_logger
import sys
import pickle


def __get_ipython_func():
    return sys.modules['__main__'].__dict__.get('get_ipython', None)


def __is_in_ipython():
    get_ipython = __get_ipython_func()
    if get_ipython is None:
        return False
    else:
        return get_ipython() is not None


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


def dump_pickle(obj, path: str, protocol=pickle.HIGHEST_PROTOCOL):
    f = open(path + '.pickle', 'wb')
    pickle.dump(obj, f, protocol=protocol)
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
