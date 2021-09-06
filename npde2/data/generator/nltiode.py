"""
Non-Linear Time-Invariant ODE
"""

import numpy

from npde2.data.generator.data_generator import DataGenerator


class NLTIODE(DataGenerator):
    def __init__(self, parameters):
        super(NLTIODE, self).__init__(parameters)

    def generate(self, Ny: list[int], model: str) -> list[numpy.array]:
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
        if model == 'vdp':
            pass
        else:
            raise ValueError(f"model {model} is not supported for data generation")

    def __vdp(self, Ny: list[int], x0: list[numpy.array], t_end: float) -> list[numpy.array]:

        """
        Generate multiple-series dataset following the model for VDP oscillator
        @param Ny: list[int] list of integers , each for the length of each series
        @param x0: list[numpy-array] : list of d-dimensional array, each of which represent the initial condition
            for each series
        @param t_end: float : float representing the value of the time-end for the series
        @return: list[numpy.array] : list of numpy array, each of which of dimension d, represents a separate series
        """
        x0 = self.parameters['x0']

        pass
