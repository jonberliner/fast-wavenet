import tensorflow as tf
tfd = tf.contrib.distributions
# import sys
# sys.path.append('./')
# import pdb
# pdb.set_trace()
import models.tf_modules as tfm
import util
import numpy as np

def gmvae(DX, DY, DZ_NORMAL, DZ_BERNOULLI, D_HID, P_DROP, BN):
    loss = tf.constant(0.)
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

        qybernoulli_logit = tfm.linear(h3, DY, name='qzbl')
        qy_prob = tf.nn.sigmoid(qybernoulli_logit)
        qy = tfd.Bernoulli(logits=qybernoulli_logit)


    ysample = y_ph
    yvs = 'y'
    with tf.variable_scope('pz_given_%s' % yvs) as scope:
        pznormal_mu = tfm.linear(ysample, DZ_NORMAL, name='qznm')
        pznormal_lv = tfm.linear(ysample, DZ_NORMAL, name='qznlv')
        pznormal = tfd.Normal(pznormal_mu, tf.nn.softplus(pznormal_lv))

        pzbernoulli_logit = tfm.linear(ysample, DZ_BERNOULLI, name='qzbl')
        pzbernoulli = tfd.Bernoulli(logits=pzbernoulli_logit)


    with tf.variable_scope('qz_given_%s' % yvs) as scope:
        qz_inference_input = tf.concat([xsample, ysample], 1)
        h1 = tfm.fc_layer(qz_inference_input, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h1', is_training=is_training_ph)

        h20 = tfm.fc_layer(h1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h20', is_training=is_training_ph)
        h2 = h20

        h30 = tfm.fc_layer(h1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h3', is_training=is_training_ph)
        h3 = h30

        qznormal_mu = tfm.linear(h3, DZ_NORMAL, name='qznm')
        qznormal_lv = tfm.linear(h3, DZ_NORMAL, name='qznlv')
        qznormal_var = tf.nn.softplus(qznormal_lv + 1e-6)
        qznormal = tfd.Normal(qznormal_mu, qznormal_var)
        znormal = qznormal.sample()

        qzbernoulli_logit = tfm.linear(h3, DZ_BERNOULLI, name='qzbl')
        qzbernoulli = tfd.Bernoulli(logits=qzbernoulli_logit)
        rqzbernoulli = tfd.RelaxedBernoulli(temperature=temp_ph, logits=qzbernoulli_logit)
        zbernoulli = rqzbernoulli.sample()
        zbernoulli.set_shape((None, DZ_BERNOULLI))


    with tf.variable_scope('px_given_%s' % yvs) as scope:
        # TODO: make discrete work
        zsample = tf.concat([znormal, zbernoulli], 1)
        # zsample = znormal
        dh1 = tfm.fc_layer(zsample, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='dh1', is_training=is_training_ph)

        dh20 = tfm.fc_layer(dh1, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='dh20', is_training=is_training_ph)
        dh2 = dh20

        dh30 = tfm.fc_layer(dh2, dim_output=D_HID, act_fn=tf.nn.relu, p_drop=P_DROP, batch_norm=BN, name='h30', is_training=is_training_ph)
        dh3 = dh30

        px_logits = tfm.linear(dh3, DX)
        px = tfd.Bernoulli(logits=px_logits+1e-6)


    with tf.name_scope('losses_given_%s' % yvs) as scope:
        py = tfd.Bernoulli(probs=tf.clip_by_value(y_ph, 0.01, 0.99))
        kl_qp_y = tfd.kl_divergence(qy, py)
        kl_qp_z_normal = tfd.kl_divergence(qznormal, pznormal)
        kl_qp_z_bernoulli = tfd.kl_divergence(qzbernoulli, pzbernoulli)
        negloglik_x = -px.log_prob(xsample)
        # negloglik_x = tf.nn.sigmoid_cross_entropy_with_logits(logits=px_logits,
        #                                                       labels=x_ph)

        losses_given_y = tf.reduce_sum(kl_qp_y, 1) +\
                         tf.reduce_sum(negloglik_x, 1) +\
                         tf.reduce_sum(kl_qp_z_normal, 1) +\
                         tf.reduce_sum(kl_qp_z_bernoulli, 1)

            # if y0 == 0.:
            #     weighted_losses_given_y = losses_given_y * (1.-qy_prob)
            # else:
            #     weighted_losses_given_y = losses_given_y * qy_prob

        weighted_losses_given_y = losses_given_y
        loss += tf.reduce_mean(weighted_losses_given_y)

        train_loss = tf.identity(loss)
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

    out.qznormal_mu = qznormal_mu
    out.qznormal_var = tf.nn.softplus(qznormal_lv)

    out.qzbernoulli_prob = tf.nn.sigmoid(qzbernoulli_logit)

    out.negloglik_x = negloglik_x

    return out

