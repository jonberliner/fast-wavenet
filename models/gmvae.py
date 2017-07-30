import tensorflow as tf
tfd = tf.contrib.distributions
# import sys
# sys.path.append('./')
# import pdb
# pdb.set_trace()
import models.tf_modules as tfm
import util
import numpy as np

clip_logit = lambda logit, vmin=-10., vmax=10: tf.clip_by_value(logit, vmin, vmax)

def gmvae(DX, DY, DZ_NORMAL, DZ_BERNOULLI, D_HID, P_DROP, BN):
    with tf.variable_scope('placeholders') as scope:
        x_ph = tfm.ph((None, DX))
        bs = tf.shape(x_ph)[0]
        y_ph = tfm.ph((None, DY))
        temp_ph = tfm.ph(None)
        is_training_ph = tfm.ph(None, tf.bool)


    with tf.variable_scope('qy') as scope:
        xsample = x_ph
        # xsample = tf.to_float(tfd.Bernoulli(probs=x_ph).sample())
        h1 = tfm.fc_layer(xsample, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h1', is_training=is_training_ph)

        h20 = tfm.fc_layer(h1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h20', is_training=is_training_ph)
        h2 = h20  # res

        h30 = tfm.fc_layer(h1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h3', is_training=is_training_ph)
        h3 = h30  # res

        qybernoulli_logit = clip_logit(tfm.linear(h3, DY, name='qybl'))
        qy_prob = tf.nn.sigmoid(qybernoulli_logit)
        qy = tfd.Bernoulli(logits=qybernoulli_logit)

    qy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')


    ysample = y_ph
    yvs = 'y'
    EPS = 1e-4
    with tf.variable_scope('pz') as scope:
        pznormal_mu = tfm.linear(ysample, DZ_NORMAL, name='qznm')
        pznormal_lv = tfm.linear(ysample, DZ_NORMAL, name='qznlv')
        pznormal_var = tf.nn.softplus(pznormal_lv) + EPS
        pznormal = tfd.Normal(pznormal_mu, pznormal_var)

        pzbernoulli_logit = clip_logit(tfm.linear(ysample, DZ_BERNOULLI, name='qzbl'))
        pzbernoulli = tfd.Bernoulli(logits=pzbernoulli_logit)

    pz_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pz')


    with tf.variable_scope('qz') as scope:
        qz_inference_input = tf.concat([xsample, ysample], 1)
        h1 = tfm.fc_layer(qz_inference_input, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h1', is_training=is_training_ph)

        h20 = tfm.fc_layer(h1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h20', is_training=is_training_ph)
        h2 = h20

        h30 = tfm.fc_layer(h1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h3', is_training=is_training_ph)
        h3 = h30

        qznormal_mu = tfm.linear(h3, DZ_NORMAL, name='qznm')
        qznormal_lv = tfm.linear(h3, DZ_NORMAL, name='qznlv')
        qznormal_var = tf.nn.softplus(qznormal_lv) + EPS
        qznormal = tfd.Normal(qznormal_mu, qznormal_var)
        znormal = qznormal.sample()

        qzbernoulli_logit = clip_logit(tfm.linear(h3, DZ_BERNOULLI, name='qzbl'))
        qzbernoulli = tfd.Bernoulli(logits=qzbernoulli_logit)
        rqzbernoulli = tfd.RelaxedBernoulli(temperature=temp_ph, logits=qzbernoulli_logit)
        zbernoulli = rqzbernoulli.sample()
        zbernoulli.set_shape((None, DZ_BERNOULLI))

    qz_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')


    with tf.variable_scope('px') as scope:
        # TODO: make discrete work
        # zsample = tf.concat([znormal, zbernoulli], 1)
        zsample = znormal
        dh1 = tfm.fc_layer(zsample, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='dh1', is_training=is_training_ph)

        dh20 = tfm.fc_layer(dh1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='dh20', is_training=is_training_ph)
        dh2 = dh20

        dh30 = tfm.fc_layer(dh2, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h30', is_training=is_training_ph)
        dh3 = dh30

        px_alpha_logits = tfm.linear(dh3, DX, name='px_alpha_logits')
        px_alpha = tf.nn.softplus(px_alpha_logits) + EPS
        px_beta_logits = tfm.linear(dh3, DX, name='px_beta_logits')
        px_beta = tf.nn.softplus(px_beta_logits) + EPS

        px = tfd.Beta(px_alpha, px_beta)
    px_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')


    with tf.name_scope('losses_given_%s' % yvs) as scope:
        # py = tfd.Bernoulli(probs=tf.clip_by_value(y_ph, 0.01, 0.99))
        # kl_qp_y = tfd.kl_divergence(qy, py)
        kl_qp_y = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=qybernoulli_logit,
                    labels=y_ph)
        kl_qp_z_normal = tfd.kl_divergence(qznormal, pznormal)
        kl_qp_z_bernoulli = tfd.kl_divergence(qzbernoulli, pzbernoulli)
        negloglik_x = -px.log_prob(xsample)

        losses = tf.reduce_sum(kl_qp_y, 1) +\
                 tf.reduce_sum(negloglik_x, 1) +\
                 tf.reduce_sum(kl_qp_z_normal, 1) +\
                 tf.reduce_sum(kl_qp_z_bernoulli, 1)

        loss = tf.reduce_mean(losses)

        train_loss = tf.identity(loss)
        # add elastic penalty (l1 + l2) to linear transform params
        WL1 = 1e-3
        WL2 = 1e-3
        l1 = lambda v: tf.reduce_sum(tf.abs(v))
        l2 = lambda v: tf.reduce_sum(tf.square(v))
        elastic = lambda v, wl1, wl2: l1(v)*wl1 + l2(v)*wl2

        for var in qy_vars + pz_vars + qz_vars + px_vars:
            if 'weights' in var.name or 'bias' in var.name:
                train_loss += elastic(var, WL1, WL2)

        trainer = tf.train.AdamOptimizer(1e-3).minimize(train_loss)

    out = util._Bunch()
    out.sess = tf.Session()
    out.sess.run(tf.global_variables_initializer())

    out.x_ph = x_ph
    out.y_ph = y_ph
    out.temp_ph = temp_ph
    out.is_training_ph = is_training_ph

    out.loss = loss
    out.label_loss = tf.reduce_mean(kl_qp_y)
    out.train_loss = train_loss

    out.trainer = trainer


    out.qy_prob = qy_prob

    out.qznormal_mu = qznormal_mu
    out.qznormal_var = tf.nn.softplus(qznormal_lv)

    out.qzbernoulli_prob = tf.nn.sigmoid(qzbernoulli_logit)

    out.negloglik_x = negloglik_x

    return out

