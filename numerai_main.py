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

# # REAL PERFORMER: do era holdout for validation
# holdouts = rng.choice(np.unique(eras), 5, replace=False).tolist()
# i_train = np.where(np.vectorize(lambda string: string not in holdouts)(eras))[0]
# i_test = np.where(np.vectorize(lambda string: string in holdouts)(eras))[0]

#
# randomly cut up a single era as our data
holdouts = rng.choice(np.unique(eras), 3 + 1, replace=False).tolist()
P_TRAIN = 0.9
i_all = rng.permutation(np.where(np.vectorize(lambda string: string in holdouts[:-1])(eras))[0])
n_train = int(len(i_all)*P_TRAIN)
i_train = i_all[:n_train]
i_test = i_all[n_train:]
i_holdout = rng.permutation(np.where(np.vectorize(lambda string: string in holdouts[-1])(eras))[0])


# PUT DATA IS STANDARD DATASET FORMAT
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
# holdout set
dat.holdout = util._Bunch()
dat.holdout.x = inputs[i_holdout]
dat.holdout.y = targets[i_holdout]
assert array_rank(dat.holdout.x) == 2
assert array_rank(dat.holdout.y) == 2
dat.holdout.num_examples = dat.holdout.x.shape[0]
assert dat.holdout.y.shape[0] == dat.holdout.num_examples


# BUILD OUR MODEL
# GMVAE MODEL
DX = inputs.shape[1]
DY = targets.shape[1]
P_DROP = 0.
BN = True
DZ_NORMAL = 61
DZ_BERNOULLI = 37
D_HID = 1024
model = gmvae(DX, DY, DZ_NORMAL, DZ_BERNOULLI, D_HID, P_DROP, BN)

# # DISCRIM MODEL (called res but mess around with a buncha variants on a big ol mlp regression)
# DX = inputs.shape[1]
# DY = targets.shape[1]
# D_HID = 256
# N_HID = 10
# P_DROP = 0.#2
# BN = False
# model = res(DX, DY, D_HID, N_HID, P_DROP, BN)

def _prep_fd(d0, i_batch, temp0, is_training):
    x0 = d0.x[i_batch, :]
    y0 = d0.y[i_batch, :]
    return {model.x_ph: x0,
            model.y_ph: y0,
            model.temp_ph: temp0,
            model.is_training_ph: is_training}


def _train_step(sess, d0, i_batch, temp0, is_training, i_step):
    fd = _prep_fd(d0, i_batch, temp0, is_training)
    _, l0 = sess.run([model.trainer, model.label_loss], feed_dict=fd)
    print('label_loss step %d: %.3f' % (i_step, l0), end='\r')


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
        l0, py0 = sess.run([model.label_loss, model.qy_prob], feed_dict=fd)
        acc0 = ((py0 > 0.5) == fd[model.y_ph]).sum()
        loss0.append(l0)
        acc += acc0

        i_start = i_end
    acc /= nn
    loss0 = np.mean(loss0)
    print('\n%s label_loss step %d: %.3f' % (split_name, i_step, loss0))
    print('%s accuracy step %d: %.3f\n' % (split_name, i_step, acc))



# # training params
n_step = int(1e6)
BATCH_SIZE = 128
TEST_EVERY = int(1e3)
# params for discrete optimization only
ANNEAL_EVERY = int(1e2)
ANNEAL_RATE = 0.9
INIT_TEMP = 2.

# let's train!
temp0 = INIT_TEMP
batcher = util.Batcher(dat.train.num_examples, BATCH_SIZE)
for i_step in range(n_step):
    i_batch = batcher()
    _train_step(model.sess, dat.train, i_batch, temp0, True, i_step)
    if i_step % TEST_EVERY == 0:
        _test(model.sess, dat.train, 1e-5, False, i_step, 'train')
        _test(model.sess, dat.test, 1e-5, False, i_step, 'test')
        _test(model.sess, dat.holdout, 1e-5, False, i_step, 'holdout')
    if i_step % ANNEAL_EVERY == 0:
        temp0 *= ANNEAL_RATE

