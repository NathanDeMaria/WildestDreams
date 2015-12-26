import os
import lmdb
import PIL.Image
import numpy as np
import caffe
from sklearn.cross_validation import StratifiedShuffleSplit


def _to_numpy(image, pixels):
    """
    Goes from filename to numpy array
    :param image: filename of image
    :param pixels: number of pixels to rescale image to
    :return:
    """
    img = PIL.Image.open(image)
    return np.float32(img.resize((pixels, pixels)).convert('RGB'))


def read_data(images, pixels, teams, cap=1000):
    """
    Reads all the images I have, and their team labels
    :param images: list of paths to images
    :param pixels: number of pixels to rescale image to
    :param cap: max # of images to include
    :param teams: list of teams
    :return: tuple(input=images array IMAGESxPIXELSxPIXELSxCHANNELS(3), labels=labels array IMAGESxTEAMS)
    """
    files = images[:cap]

    images = np.array([_to_numpy(f, pixels) for f in files])
    labels = [os.path.basename(f).split('_')[0] for f in files]

    return images, [teams.index(t) for t in labels]


def format_data(images):
    """
    Reshapes and scales the images
    :param images: np array IMAGESxPIXELSxPIXELSx3
    :return: np array IMAGESx3xPIXELSxPIXELS
    """
    X = np.rollaxis(images, 3, 1)

    for i in xrange(X.shape[0]):
        mu, std = X[i, :, :, :].mean(), X[i, :, :, :].std()
        rescaled_slice = (X[i, :, :, :] - mu) / std
        X[i, :, :, :] = rescaled_slice

    return X


def write_lmdb(lmdb_name, X, y):
    """
    Write the data to lmdb database
    :param lmdb_name: name of the lmdb database
    :param X: features
    :param y: labels
    :return: None
    """
    map_size = X.nbytes * 10

    env = lmdb.open(lmdb_name, map_size=map_size)

    with env.begin(write=True) as txn:
        for i in range(X.shape[0]):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


def split_train_test(X, y):
    """
    Splits the data into a train and test set, preserving proportions of each class
    :param X: features
    :param y: labels
    :return: X_train, y_train, X_test, y_test
    """
    # for loop, but there's only one :)
    for trains, tests in StratifiedShuffleSplit(y, 1, test_size=.3):
        X_train = X[trains, :, :, :]
        y_train = y[trains]
        X_test = X[tests, :, :, :]
        y_test = y[tests]
        return X_train, y_train, X_test, y_test
