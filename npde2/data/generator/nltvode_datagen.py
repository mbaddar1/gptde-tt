"""
Non-Linear Time-Variant ODE

two examples
1)Forced Van der Pol oscillator http://math.colgate.edu/~wweckesser/pubs/FVDPI.pdf (eq 1.1 1.2)
2) https://arxiv.org/pdf/1512.02302.pdf eq (33)
3) https://downloads.hindawi.com/journals/mpe/2011/749309.pdf (eq 7.1)

"""
import logging
import typing

import numpy as np
import scipy.stats
from scipy.integrate import solve_ivp

from npde2.data.generator.de_generator import DiffEqDataGen


class NLTVODE(DiffEqDataGen):
    def __init__(self, model):
        """

        @param parameters:
        """
        super().__init__(model)

    def generate(self, N: int, **kwargs) -> typing.Tuple:
        if self.model == 'vdp-forced':  # vdp forced
            return NLTVODE.__vdp_forced_generate(N=N, t_span=kwargs['t_span'], y0=kwargs['y0'], mio=kwargs['mio'],
                                                 omega=kwargs['omega'], a=kwargs['a'],nstd=kwargs['nstd'])
        else:
            raise ValueError(f'Model {self.model} is not supported!')

    def plot(self):
        pass

    @staticmethod
    def __vdp_forced_generate(N, t_span, y0, mio, omega, a, nstd):
        t_eval = np.linspace(start=t_span[0], stop=t_span[1], num=N)
        soln = solve_ivp(fun=NLTVODE.__vdp_forced_func, t_span=t_span, y0=y0, t_eval=t_eval, args=(mio, omega, a))
        y = soln.y
        y += scipy.stats.norm.rvs() * nstd
        return y

    @staticmethod
    def __vdp_forced_func(t, y, mio, omega, a):
        """
        ref:
        https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Forced_Van_der_Pol_oscillator
        numerical example
        https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#/media/File:Vanderpol_time_mu=8.53_A=1.2_T=10.svg
        @param t:
        @param y:
        @param mio:
        @return:
        """
        y_dot = np.zeros(shape=(2,))
        y_dot[0] = y[1]
        y_dot[1] = mio * (1 - y[0] ** 2) * y[1] - y[0] + a * np.sin(omega * t)
        return y_dot


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('NLTVODE-Main')
    nltvode = NLTVODE(model='vdp-forced')

    kwargs = {'t_span': (0, 8), 'y0': [0.2, 0.4], 'mio': 8.53, 'a': 1.2, 'omega': 2.0 * np.pi / 10}
    soln = nltvode.generate(N=10, **kwargs)
    logger.info(f'Soln = {soln}')
