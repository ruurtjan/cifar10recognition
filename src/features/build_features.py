import logging
import os
import pickle

import numpy as np

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def main():
    """
        Normalizes the given feature data sets and saves them in processed folder.
    """
    logger.info('Normalize data')
    with open(os.path.join(os.getcwd(), "data", "interim", "data_dictionary.pickle"), "rb") as pickle_file:
        data_dict = pickle.load(pickle_file)
    data_dict['train_X'], data_dict['val_X'], data_dict['test_X'], mean, std = normalize_all_data(data_dict['train_X'],
                                                                                                  data_dict['val_X'],
                                                                                                  data_dict['test_X'])
    with open(os.path.join(os.getcwd(), "data", "processed", "data_dictionary.pickle"), "wb") as pickle_file:
        pickle.dump(data_dict, pickle_file)

    extra_info_dict = {'mean': mean, 'std': std, 'label_to_names': data_dict['label_to_names']}
    with open(os.path.join(os.getcwd(), "models", "extra_info_dictionary.pickle"), "wb") as pickle_file:
        pickle.dump(extra_info_dict, pickle_file)


def normalize_all_data(train, val, test):
    """
    Normalizes train, validation and test feature sets, by subtracting the standard deviation

    :param train: train feature data set
    :param val: validation feature data set
    :param test: test feature dataset
    :return: return normalized train, val, test feature data sets
    """
    mean = np.mean(train)
    std = np.std(train)
    train_X = normalize(train, mean, std)
    val_X = normalize(val, mean, std)
    test_X = normalize(test, mean, std)

    return train_X, val_X, test_X, mean, std


def normalize(data, mean, std):
    """
    Normalizes data
    :param data: numpy array feature data set
    :param mean: mean of train feature data set
    :param std: standard deviation of train feature data set
    :return: normalized data set
    """
    return (data - mean) / std


if __name__ == '__main__':
    main()
