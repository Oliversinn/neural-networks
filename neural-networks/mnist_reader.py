import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from the given directory path.

    Parameters
    ----------
    path : str
        Directory containing the MNIST .gz files.
    kind : str
        Either 'train' or 't10k' (test).

    Returns
    -------
    images : np.ndarray, shape (n_samples, 784)
    labels : np.ndarray, shape (n_samples,)
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels
