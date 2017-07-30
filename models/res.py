import tensorflow as tf
tfd = tf.contrib.distributions
import models.tf_modules as tfm
import util
import numpy as np

def res(DX, DY, D_HID, N_HID, P_DROP, BN):
    loss = tf.constant(0.)
    with tf.variable_scope('placeholders') as scope:
        x_ph = tfm.ph((None, DX))
        y_ph = tfm.ph((None, DY))
        is_training_ph = tfm.ph(None, tf.bool)
        temp_ph = tfm.ph(None)

    with tf.variable_scope('inference') as scope:
        xsample = x_ph

        out = tfm.fc_layer(xsample, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h0', is_training=is_training_ph)
        # hids.append(out)

        for hi in range(1, N_HID):
            res_hi = tfm.fc_layer(out,
                                  dim_output=D_HID,
                                  act_fn=tf.nn.relu,
                                  p_drop=0.,
                                  batch_norm=BN,
                                  name='h%d_res' % hi,
                                  is_training=is_training_ph)
            out = res_hi  # + out
            out = tfm.dropout(out, is_training=is_training_ph, p_drop=P_DROP)

        yhat_logit = tfm.linear(out, DY, name='yhat_logit')
        qy_prob = tf.nn.sigmoid(yhat_logit)


    with tf.name_scope('loss') as scope:
        l1 = lambda v: tf.reduce_sum(tf.abs(v))
        l2 = lambda v: tf.reduce_sum(tf.square(v))
        elastic = lambda v, wl1, wl2: l1(v)*wl1 + l2(v)*wl2
        WL1 = 1e-5
        WL2 = 1e-5

        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat_logit, labels=y_ph)
        loss = tf.reduce_mean(losses)

        train_loss = tf.identity(loss)
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='inference'):
        #     if 'weights' or 'bias' in var.name:
        #         train_loss += elastic(var, WL1, WL2)
    trainer = tf.train.AdamOptimizer(1e-3).minimize(train_loss)

    out = util._Bunch()
    out.sess = tf.Session()
    out.sess.run(tf.global_variables_initializer())

    out.x_ph = x_ph
    out.y_ph = y_ph
    out.temp_ph = temp_ph
    out.is_training_ph = is_training_ph

    out.loss = loss
    out.train_loss = train_loss

    out.trainer = trainer

    out.qy_prob = qy_prob

    return out


