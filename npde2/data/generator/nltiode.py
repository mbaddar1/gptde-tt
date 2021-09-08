"""
Non-Linear Time-Invariant ODE
"""
import logging
import typing

import numpy as np
from scipy.integrate import solve_ivp

from npde2.data.generator.de_generator import DiffEqDataGen


class NLTIODE(DiffEqDataGen):
    def __init__(self, model):
        super(NLTIODE, self).__init__(model)

    def generate(self, N: int, **kwargs) -> typing.Tuple:
        """

        @param model: str : model type to generate the data
            can be one of ["vdp"]
        @param Ny: list[int] list of lengths for the multiple series dataset generated
        @return: list[numpy.array] list of np-arrays, each represents a series of the data
            for example , data with 3-series
                y1 => first series of len N1
                y2 => 2nd series of len N2
                y3 => 3rd series of len N3
                each of which of dimension D, which is inferred from model type
                for ex. current implementation for VDP is for 2-dim case
        """
        if self.model == 'vdp':
            soln = NLTIODE.__vdp_generate(N=N, t_span=kwargs['t_span'], y0=kwargs['y0'], mio=kwargs['mio'])
            return soln
        else:
            raise ValueError(f"model {self.model} is not supported for data generation")

    @staticmethod
    def __vdp_func(t, y, mio):
        """
        differential function for Van der Pol in 2-d form
        ref:
        https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
        @param y:
        @param t:
        @param mio:
        @return:
        """
        y_dot = np.zeros(shape=(2,))
        y_dot[0] = y[1]
        y_dot[1] = mio * (1 - y[0] ** 2) * y[1] - y[0]
        return y_dot

    @staticmethod
    def __vdp_generate(N, t_span, y0, mio) -> typing.Tuple:

        """
        Generate multiple-series dataset following the model for VDP oscillator
        @param Ny: list[int] list of integers , each for the length of each series
        @param x0: list[numpy-array] : list of d-dimensional array, each of which represent the initial condition
            for each series
        @param t_end: float : float representing the value of the time-end for the series
        @return: list[numpy.array] : list of numpy array, each of which of dimension d, represents a separate series
        """
        # t_span = (0, 8)
        t_eval = np.linspace(start=t_span[0], stop=t_span[1], num=10)
        # y0 = np.array([0.2, 0.4])
        # mio = 1
        soln = solve_ivp(fun=NLTIODE.__vdp_func, t_span=t_span, t_eval=t_eval, y0=y0, args=(mio,))
        return soln

    def plot(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('NLTIODE-Main')
    nltiode = NLTIODE(model='vdp')
    kwargs = {'t_span': (0, 8), 'y0': [0.2, 0.4], 'mio': 8.53}
    soln = nltiode.generate(N=10, **kwargs)
    logger.info(f'Soln = \n{soln}')
