import numpy as np
import pandas as pd
import os
rng = np.random

class Batcher(object):
    def __init__(self, X, batch_size, i_start=0, random_order=True):
        if type(X) == int:
            self.N = X
            self.X = np.arange(self.N)
        else:
            self.N = X.shape[0]
            self.X = X

        self.random_order = random_order
        self.order = np.arange(self.N)
        if self.random_order:
            rng.shuffle(self.order)
        self.batch_size = batch_size
        self.i_start = i_start
        self.get_i_end = lambda: min(self.i_start + self.batch_size, self.N)

        self.end_of_epoch = lambda: self.i_start == self.N
        self.batch_inds = None

    def __call__(self):
        inds = self.next_inds()
        return self.X[inds]

    def next_inds(self):
        i_end = self.get_i_end()
        if self.i_start == i_end:
            if self.random_order:
                rng.shuffle(self.order)
            self.i_start = 0
            i_end = self.get_i_end()
        batch_inds = self.order[self.i_start:i_end]
        batch_inds.sort()
        # increment
        self.i_start = i_end
        self.batch_inds = batch_inds
        return batch_inds


class Bunch(dict):
    """dicts with dot notation access.  also needed to make datasets swappable with tensorflow
    example datasets for quick swapping between test data and our data"""
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


def make_numerai_batch(path):
    df = pd.read_csv(path)
    df_input_cols= [df[feature].values for feature in df.keys() if 'feature' in feature]
    inputs = np.vstack(df_input_cols).T

    targets = df.target.values[:, None]

    eras = df.era.values

    return inputs, targets, eras


def mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)


def get_mnist(classes=None, n_per_class=None, n_per_class_test=None, onehot_label=True, seed=None, data_dir=''):
    classes = classes or np.arange(10).tolist()
    n_class = len(classes)
    for li in classes:
        assert li in np.arange(10)

    if n_per_class:
        if type(n_per_class) == list:
            assert len(n_per_class) == n_class
        else:
            n_per_class = [n_per_class]*n_class
    if n_per_class_test:
        if type(n_per_class_test) == list:
            assert len(n_per_class_test) == n_class
        else:
            n_per_class_test = [n_per_class_test] * n_class

    from tensorflow.examples.tutorials.mnist import input_data
    seed = seed or np.random.randint(1e8)
    rng = np.random.RandomState(seed)
    mnist = input_data.read_data_sets(os.path.join(data_dir, "MNIST_data/"), one_hot=False)
    dat = Bunch()
    dat.train = Bunch()
    dat.test = Bunch()
    dat.seed = seed


    dat.train.images = [None] * n_class
    dat.test.images = [None] * n_class
    dat.train.labels = [None] * n_class
    dat.test.labels = [None] * n_class
    for li, lab in enumerate(classes):
        i_train = np.where(mnist.train.labels == lab)[0]
        i_test = np.where(mnist.test.labels == lab)[0]
        dat.train.images[li] = mnist.train.images[i_train]
        dat.train.labels[li] = mnist.train.labels[i_train]
        dat.test.images[li] = mnist.test.images[i_test]
        dat.test.labels[li] = mnist.test.labels[i_test]

        if n_per_class:
            i_lab = rng.choice(dat.train.images[li].shape[0], n_per_class[li], replace=False)
            dat.train.images[li] = dat.train.images[li][i_lab]
            dat.train.labels[li] = dat.train.labels[li][i_lab]

        if n_per_class_test:
            i_lab = rng.choice(dat.test.images[li].shape[0], n_per_class_test[li], replace=False)
            dat.test.images[li] = dat.test.images[li][i_lab]
            dat.test.labels[li] = dat.test.labels[li][i_lab]

    dat.train.images = np.concatenate(dat.train.images, 0)
    dat.train.labels = np.concatenate(dat.train.labels, 0)
    dat.test.images = np.concatenate(dat.test.images, 0)
    dat.test.labels = np.concatenate(dat.test.labels, 0)
    dat.train.num_examples = dat.train.images.shape[0]
    dat.test.num_examples = dat.test.images.shape[0]

    if onehot_label:
        # TODO: should we leave as 10 classes or shrink?
        dat.train.labels = onehot(10, dat.train.labels)
        dat.test.labels = onehot(10, dat.test.labels)

    return dat


onehot = lambda n, ii: np.eye(n)[ii]
