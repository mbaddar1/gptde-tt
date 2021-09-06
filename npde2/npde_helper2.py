import logging

import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils import plot_model

tf.disable_v2_behavior()

from kernels import OperatorKernel
from npde2.npde_models2 import NPODE2


def build_model_2(sess, t, Y, model='ode', sf0=1.0, ell0=[2, 2], sfg0=1.0, ellg0=[1e5],
                  W=6, ktype="id", whiten=True,
                  fix_ell=False, fix_sf=False, fix_Z=False, fix_U=False, fix_sn=False,
                  fix_ellg=False, fix_sfg=False, fix_Zg=True, fix_Ug=False):
    """
    Args:
        sess: TensowFlow session needed for initialization and optimization
        t: Python array of numpy vectors storing observation times
        Y: Python array of numpy matrices storing observations. Observations
             are stored in rows.
        model: 'sde' or 'ode'
        sf0: Integer initial value of the signal variance of drift GP
        ell0: Python/numpy array of floats for the initial value of the
            lengthscale of drift GP
        sfg0: Integer initial value of the signal variance of diffusion GP
        ellg0: Python/numpy array of a single float for the initial value of the
            lengthscale of diffusion GP
        W: Integer denoting the width of the inducing point grid. If the problem
            dimension is D, total number of inducing points is W**D
        ktype: Kernel type. We have made experiments only with Kronecker kernel,
            denoted by 'id'. The other kernels are not supported.
        whiten: Boolean. Currently we perform the optimization only in the
            white domain
        fix_ell: Boolean - whether drift GP lengthscale is fixed or optimized
        fix_sf: Boolean - whether drift GP signal variance is fixed or optimized
        fix_Z: Boolean - whether drift GP inducing locations are fixed or optimized
        fix_U: Boolean - whether drift GP inducing vectors are fixed or optimized
        fix_sn: Boolean - whether noise variance is fixed or optimized
        fix_ellg: Boolean - whether diffusion GP lengthscale is fixed or optimized
        fix_sfg: Boolean - whether diffusion GP signal variance is fixed or optimized
        fix_Zg: Boolean - whether diffusion GP inducing locations are fixed or optimized
        fix_Ug: Boolean - whether diffusion GP inducing vectors are fixed or optimized
    Returns:
        npde: A new NPDE model
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('Build-Model')
    logger.info('Model being initialized...')

    def init_U0(Y=None, t=None, kern=None, Z0=None, whiten=None):
        # Ug is approximating the gradient using lag 1 different
        # Ug[t(n)] = (  Y(t(n+1)) - Y(t(n)) )/ ( t(n+1) - t(n) )
        Ug = (Y[1:, :] - Y[:-1, :]) / np.reshape(t[1:] - t[:-1], (-1, 1))
        with tf.name_scope("init_U0"):
            tmp = NPODE2(Z0=Y[:-1, :], U0=Ug, sn0=0, kern=kern, jitter=0.25, whiten=False,
                         fix_Z=True, fix_U=True, fix_sn=True)
        U0 = tmp.f(X=Z0)
        if whiten:
            Lz = tf.cholesky(kern.K(Z0))
            U0 = tf.matrix_triangular_solve(Lz, U0, lower=True)
        U0 = sess.run(U0)
        return U0

    # start of model building
    D = len(ell0)
    Nt = len(Y)
    x0 = np.zeros((Nt, D))
    # stack all Nt multiples series along each dimension, that target is to find the min/max values along each dimension
    # over the multiple-series
    Ys = np.zeros((0, D)) # Y_stack ?
    for i in range(Nt):
        x0[i, :] = Y[i][0, :]
        Ys = np.vstack((Ys, Y[i]))
    maxs = np.max(Ys, 0) # find max values along each dimension in the stacked multiple-series
    mins = np.min(Ys, 0) # find the min too
    grids = []
    for i in range(D):
        grids.append(np.linspace(mins[i], maxs[i], W))
    vecs = np.meshgrid(*grids)
    Z0 = np.zeros((0, W ** D))
    for i in range(D):
        Z0 = np.vstack((Z0, vecs[i].T.flatten()))
    Z0 = Z0.T

    tmp_kern = OperatorKernel(sf0, ell0, ktype="id", fix_ell=True, fix_sf=True)

    U0 = np.zeros(Z0.shape, dtype=np.float64)
    for i in range(len(Y)):
        U0 += init_U0(Y[i], t[i], tmp_kern, Z0, whiten)
    U0 /= len(Y) # averaging over multiple-series

    sn0 = 0.5 * np.ones(D)
    Ug0 = np.ones([Z0.shape[0], 1]) * 0.01

    ell0 = np.asarray(ell0, dtype=np.float64)
    ellg0 = np.asarray(ellg0, dtype=np.float64)

    kern = OperatorKernel(sf0=sf0, ell0=ell0, ktype=ktype, fix_ell=fix_ell, fix_sf=fix_sf)

    if model is 'ode':
        npde2 = NPODE2(Z0=Z0, U0=U0, sn0=sn0, kern=kern, whiten=whiten, fix_Z=fix_Z, fix_U=fix_U, fix_sn=fix_sn)
        sess.run(tf.global_variables_initializer())
        return npde2

    elif model is 'sde':
        # diffus = BrownianMotion(sf0=sfg0, ell0=ellg0, U0=Ug0, Z0=Z0, whiten=whiten, \
        #                         fix_sf=fix_sfg, fix_ell=fix_ellg, fix_Z=fix_Zg, fix_U=fix_Ug)
        # npde = NPSDE(Z0=Z0, U0=U0, sn0=sn0, kern=kern, diffus=diffus, whiten=whiten, \
        #              fix_Z=fix_Z, fix_U=fix_U, fix_sn=fix_sn)
        # sess.run(tf.global_variables_initializer())
        # return npde
        logger.error(f'{model} not implemented in npde2 yet!')
        raise NotImplementedError(f'{model} not implemented in npde2 yet!')

    else:
        raise NotImplementedError("model parameter should be either 'ode' or 'sde', not {:s}\n".format(model))


def fit_model_2(sess, model, t, Y, Nw=10, num_iter=500, print_every=10, eta=5e-3, dec_step=20, dec_rate=0.99, plot_=True):
    """ Fits the NPDE model to a dataset and returns the fitted object

    Args:
        sess: TensowFlow session needed for initialization and optimization
        t: Python array of numpy vectors storing observation times
        Y: Python array of numpy matrices storing observations. Observations
             are stored in rows.
        Nw: Integer number of samples used for optimization in SDE model
        num_iter: Integer number of optimization steps
        num_iter: Integer interval of optimization logs to be printed
        eta: Float step size used in optimization, must be carefully tuned
        dec_step: Float decay interval of the step size
        dec_rate: Float decay rate of the step size
        plot_: Boolean for plotting the model fit. Valid only for demo

    Returns:
        npde: Fitted model
    """
    print('Building loss function...')
    x0 = np.vstack([Y_[0] for Y_ in Y])
    if model.name is 'npode':
        with tf.name_scope("cost"):
            X = model.forward(x0, t)
            ll = []
            for i in range(len(X)):
                mvn = tfd.MultivariateNormalFullCovariance(loc=X[i],covariance_matrix=tf.diag(model.sn))
                ll.append(tf.reduce_sum(mvn.log_prob(Y[i])))
            ll = tf.reduce_logsumexp(ll)
            ode_prior = model.build_prior()
            cost = -(ll + ode_prior)

    elif model.name is 'npsde':
        with tf.name_scope("cost"):
            Xs = model.forward(x0, t, Nw=Nw)
            ll = 0
            for i in range(len(Y)):
                mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=Y[i],
                                                                                covariance_matrix=tf.diag(model.sn))
                ll_i = tf.stack([mvn.log_prob(Xs[i][j, :, :]) for j in range(Xs[i].shape[0])])  # Nw x D
                ll_i = tf.reduce_sum(tf.log(tf.reduce_mean(tf.exp(ll_i), axis=0)))
                ll += ll_i
            sde_prior = model.build_prior()
            cost = -(ll + sde_prior)

    else:
        raise NotImplementedError("model parameter should be either 'ode' or 'sde', not {:s}\n".format(model))

    print('Adam optimizer being initialized...')
    with tf.name_scope("adam"):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        expdec = tf.train.exponential_decay(eta, global_step, dec_step, dec_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(expdec).minimize(cost, global_step)

    sess.run(tf.global_variables_initializer())
    # global_vars = tf.global_variables()
    # sess.run(tf.variables_initializer(var_list=[v for v in global_vars if 'adam' in v.name]))

    print('Optimization starts.')
    print('{:>16s}'.format("iteration") + '{:>16s}'.format("objective"))
    for i in range(1, num_iter + 1):
        _cost, _ = sess.run([cost, optimizer])
        if i == 1 or i % print_every == 0 or i == num_iter:
            print('{:>16d}'.format(i) + '{:>16.3f}'.format(_cost))
    print('Optimization ends.')

    if plot_:
        print('Plotting...')
        plot_model(model, t, Y, Nw=50)

    return model