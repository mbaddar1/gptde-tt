import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from npde2.npde_helper2 import build_model_2, fit_model_2
from utils import gen_data

np.random.seed(918273)  # just for illustration purposes
x0, t, Y, X, D, f, g = gen_data('vdp', Ny=[35, 40, 30], tend=8, nstd=0.1)
sess = tf.compat.v1.InteractiveSession()

npde = build_model_2(sess, t, Y, model='ode', sf0=1.0, ell0=np.ones(2), W=6, ktype="id")
npde = fit_model_2(sess, npde, t, Y, num_iter=500, print_every=50, eta=0.02, plot_=True)

