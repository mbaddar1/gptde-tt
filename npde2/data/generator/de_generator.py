"""
TODO
3 Model characteristics
i) Non-linearity => handled by GP
ii) Time-variance
iii) Time-delay
1) Generate 1 examples for Non-linear time invariant (used VDP) (NLTIODE)
2) Generate 1 Example for Non-Linear time-variant with no delay (NLTVODE)
3) Generate 1 Example for Non-Linear time-invariant with delay (NLTIDDE)
4) Generate 1 Example for Non-linear time-variant with Delay (NLTVDDE)
"""
from abc import abstractmethod, ABC


class DiffEqDataGen(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(self, N: int):
        """

        @param N: int, sample size
        @param model: model name to generate the data
            can be on of ["vdp"]
        @return: numpy-array of generated samples, the data-dimension is implicit from the model
        FIXME make the data-dimension parameterized
        """
        pass

    @abstractmethod
    def plot(self):
        pass
