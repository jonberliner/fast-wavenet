import numpy as np
import pandas as pd
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


class _Bunch(dict):
    """dicts with dot notation access.  also needed to make datasets swappable with tensorflow
    example datasets for quick swapping between test data and our data"""
    def __init__(self, *args, **kwds):
        super(_Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


def make_numerai_batch(path):
    df = pd.read_csv(path)
    df_input_cols= [df[feature].values for feature in df.keys() if 'feature' in feature]
    inputs = np.vstack(df_input_cols).T

    targets = df.target.values[:, None]

    eras = df.era.values

    return inputs, targets, eras
