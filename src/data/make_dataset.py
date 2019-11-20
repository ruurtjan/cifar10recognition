# -*- coding: utf-8 -*-
import logging
import os
import pickle
import tarfile
from urllib.request import urlretrieve

import numpy as np

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')
    data_dictionary = load_data()
    with open(os.path.join(os.getcwd(), "data", "interim", "data_dictionary.pickle"), "wb") as pickle_file:
        pickle.dump(data_dictionary, pickle_file)


def load_data():
    """
    Load cifar10 data and return this in required formats.

    :return train_X, train_y, val_X, val_y, test_X, test_y, label_to_names: train, validation, test data and labels dict
    """
    dataset_dir = download_cifar10_data()
    n_samples = 10000

    # train set
    train_X, train_y = create_train_data(dataset_dir, n_samples)

    # validation set, batch 5
    val_X, val_y = retrieve_batch(dataset_dir, "data_batch_5")

    # test set
    test_X, test_y = retrieve_batch(dataset_dir, "test_batch")

    # labels
    label_to_names = retrieve_labels(dataset_dir)

    logger.info("training set size: data = {}, labels = {}".format(train_X.shape, train_y.shape))
    logger.info("validation set size: data = {}, labels = {}".format(val_X.shape, val_y.shape))

    logger.info("Test set size: data = " + str(test_X.shape) + ", labels = " + str(test_y.shape))

    return {'train_X': train_X, 'train_y': train_y, 'val_X': val_X, 'val_y': val_y, 'test_X': test_X,
            'test_y': test_y, 'label_to_names': label_to_names}


def download_cifar10_data():
    """
    Downloads cifar10 data and places it in "data" folder

    :return dataset_dir: the filepath to the directory containing the cifar10 data
    """
    # training set, batches 1-4
    dataset_dir = os.path.join(os.getcwd(), "data", "raw")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(os.path.join(dataset_dir, "cifar-10-batches-py")):
        logger.info("Downloading data...")
        urlretrieve("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                    os.path.join(dataset_dir, "cifar-10-python.tar.gz"))
        with tarfile.open(os.path.join(dataset_dir, "cifar-10-python.tar.gz")) as tar:
            tar.extractall(dataset_dir)
    dataset_dir = os.path.join(dataset_dir, "cifar-10-batches-py")
    return dataset_dir


def create_train_data(dataset_dir, n_samples):
    """
    Load train data batches and return single train set separated in features (X) and labels (y).

    :param n_samples: number of samples in a batch
    :return train_X, train_y: the features in the x numpy array and the labels in the y numpy array for the train set
    """
    train_X = np.zeros((4 * n_samples, 3, 32, 32), dtype="float32")
    train_y = np.zeros((4 * n_samples, 1), dtype="ubyte").flatten()

    for i in range(0, 4):
        train_X[i * n_samples:(i + 1) * n_samples], train_y[i * n_samples:(i + 1) * n_samples] = \
            retrieve_batch(dataset_dir, "data_batch_" + str(i + 1))

    # Conv nets trainen duurt erg lang op CPU, dus we gebruiken maar een klein deel
    # van de data nu, als er tijd over is kan je proberen je netwerk op de volledige set te runnen
    train_X = train_X[:10000]
    train_y = train_y[:10000]

    return train_X, train_y


def retrieve_batch(dataset_dir, batchname):
    """
    Retrieves the data per batch once downloaded in download_cifar10_data function.

    :param dataset_dir: filepath containing cifar10 data
    :param batchname: name of the batch that should be retrieved
    :return x, y: the features in the x numpy array and the labels in the y numpy array
    """
    with open(os.path.join(dataset_dir, batchname), "rb") as f:
        cifar_batch = pickle.load(f, encoding="latin1")
    x = (cifar_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
    y = np.array(cifar_batch['labels'], dtype='ubyte')
    return x, y


def retrieve_labels(dataset_dir):
    """
    Retrieves label_names from meta data of cifar10 dataset. This function assumes download_cifar10_data function has
    been executed.

    :param dataset_dir: filepath containing cifar10 data
    :return label_to_names: dictionary of label to label name
    """
    with open(os.path.join(dataset_dir, "batches.meta"), "rb") as f:
        cifar_dict = pickle.load(f, encoding="latin1")
    label_to_names = {k: v for k, v in zip(range(10), cifar_dict['label_names'])}
    return label_to_names


if __name__ == '__main__':
    main()
