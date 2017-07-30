from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
rng = np.random
import pdb

from models.res import res
from models.gmvae import gmvae
import util

# GET OUR DATA
inputs, targets, eras = util.make_numerai_batch('assets/numerai/numerai_training_data.csv')
holdouts = rng.choice(np.unique(eras), 5, replace=False).tolist()

i_train = np.where(np.vectorize(lambda string: string not in holdouts)(eras))[0]
i_test = np.where(np.vectorize(lambda string: string in holdouts)(eras))[0]

holdouts = rng.choice(np.unique(eras), 1, replace=False).tolist()
P_TRAIN = 0.8
i_all = rng.permuation(np.where(np.vectorize(lambda string: string not in holdouts)(eras))[0])
n_train = int(len(i_all)*P_TRAIN)
i_train = i_all[:n_train]
i_test = i_all[n_train:]

# i_choice = rng.choice(inputs.shape[0], 100, replace=False)
# inputs = inputs[i_choice]
# targets = targets[i_choice]

array_rank = lambda npa: len(npa.shape)
dat = util._Bunch()
# train set
dat.train = util._Bunch()
dat.train.x = inputs[i_train]
dat.train.y = targets[i_train]
assert array_rank(dat.train.x) == 2
assert array_rank(dat.train.y) == 2
dat.train.num_examples = dat.train.x.shape[0]
assert dat.train.y.shape[0] == dat.train.num_examples
# test set
dat.test = util._Bunch()
dat.test.x = inputs[i_test]
dat.test.y = targets[i_test]
assert array_rank(dat.test.x) == 2
assert array_rank(dat.test.y) == 2
dat.test.num_examples = dat.test.x.shape[0]
assert dat.test.y.shape[0] == dat.test.num_examples

# BUILD OUR MODEL
N_TRAIN = inputs.shape[0]
assert targets.shape[0] == N_TRAIN
DX = inputs.shape[1]
DY = targets.shape[1]
P_DROP = 0.2
BN = False
DZ_NORMAL = 103
DZ_BERNOULLI = 19
D_HID = 101
model = gmvae(DX, DY, DZ_NORMAL, DZ_BERNOULLI, D_HID, P_DROP, BN)

# DX = inputs.shape[1]
# DY = targets.shape[1]
# D_HID = 256
# N_HID = 10
# P_DROP = 0.#2
# BN = False
# model = res(DX, DY, D_HID, N_HID, P_DROP, BN)

n_step = int(1e4)  #kwargs.get('n_step', int(1e4))
batch_size = 256  #kwargs.get('batch_size', 64)

def _prep_fd(d0, i_batch, temp0, is_training):
    x0 = d0.x[i_batch, :]
    y0 = d0.y[i_batch, :]
    return {model.x_ph: x0,
            model.y_ph: y0,
            model.temp_ph: temp0,
            model.is_training_ph: is_training}

def _train_step(sess, d0, i_batch, temp0, is_training, i_step):
    fd = _prep_fd(d0, i_batch, temp0, is_training)
    _, l0 = sess.run([model.trainer, model.loss], feed_dict=fd)
    print('loss step %d: %.3f' % (i_step, l0), end='\r')


def _test(sess, d0, temp0, is_training, i_step, split_name):
    i_start = 0
    nn = d0.num_examples
    acc = 0.
    loss0 = []
    while i_start < nn:
        print('testing split %s, %d of %d...' % (split_name, i_start, nn), end='\r')
        i_end = min(nn, i_start + 64)
        ib0 = np.arange(i_start, i_end)
        fd = _prep_fd(d0, ib0, temp0, is_training)
        l0, py0 = sess.run([model.loss, model.qy_prob], feed_dict=fd)
        acc0 = ((py0 > 0.5) == fd[model.y_ph]).sum()
        loss0.append(l0)
        acc += acc0

        i_start = i_end
    acc /= nn
    loss0 = np.mean(loss0)
    print('\n%s loss step %d: %.3f' % (split_name, i_step, loss0))
    print('%s accuracy step %d: %.3f\n' % (split_name, i_step, acc))



batcher = util.Batcher(dat.train.num_examples, batch_size)
n_step = int(1e6)
TEST_EVERY = int(1e3)
ANNEAL_EVERY = int(1e3)
ANNEAL_RATE = 0.9

temp0 = 1.
for i_step in range(n_step):
    i_batch = batcher()
    _train_step(model.sess, dat.train, i_batch, temp0, True, i_step)
    if i_step % TEST_EVERY == 0:
        _test(model.sess, dat.train, 1e-5, False, i_step, 'train')
        _test(model.sess, dat.test, 1e-5, False, i_step, 'test')
    if i_step % ANNEAL_EVERY == 0:
        temp0 *= ANNEAL_RATE

