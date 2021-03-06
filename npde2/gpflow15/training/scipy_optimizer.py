# Copyright 2017 Artem Artemev @awav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..core.compilable import Build
from ..core.errors import GPflowError
from ..models.model import Model
from . import external_optimizer, optimizer
from tensorflow.python.framework.errors_impl import InvalidArgumentError


class ScipyOptimizer(optimizer.Optimizer):
    def __init__(self, **kwargs):
        self._optimizer_kwargs = kwargs
        self._optimizer = None
        self._model = None

    def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
        """
        Make SciPy optimization tensor.
        The `make_optimize_tensor` method builds optimization tensor and initializes
        all necessary variables created by optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of variables for training.
            :param kwargs: Scipy optional optimization parameters,
                - `maxiter`, maximal number of iterations to perform.
                - `disp`, if True, prints convergence messages.
            :return: Tensorflow operation.
        """
        session = model.enquire_session(session)
        with session.as_default():
            var_list = self._gen_var_list(model, var_list)
            optimizer_kwargs = self._optimizer_kwargs.copy()
            options = optimizer_kwargs.get('options', {})
            options.update(kwargs)
            optimizer_kwargs.update(dict(options=options))
            objective = model.objective
            optimizer = external_optimizer.ScipyOptimizerInterface(objective, var_list=var_list, **optimizer_kwargs)
            return optimizer

    def minimize(self,
                 model,
                 session=None,
                 var_list=None,
                 feed_dict=None,
                 maxiter=1000,
                 disp=False,
                 initialize=False,
                 initialize_optimizer=False,
                 anchor=True,
                 step_callback=None,
                 **kwargs):
        """
        Minimizes objective function of the model.

        :param model: GPflow model with objective tensor.
        :param session: Session where optimization will be run.
        :param var_list: List of extra variables which should be trained during optimization.
        :param feed_dict: Feed dictionary of tensors passed to session run method.
        :param maxiter: Number of run interation. Note: scipy optimizer can do early stopping
            if model converged.
        :param disp: ScipyOptimizer option. Set to True to print convergence messages.
        :param initialize: If `True` model parameters will be re-initialized even if they were
            initialized before for gotten session.
        :param initialize_optimizer: If `True` model parameters will be re-initialized even if they were
            initialized before for gotten session.
        :param anchor: If `True` trained parameters computed during optimization at
            particular session will be synchronized with internal parameter values.
        :param step_callback: A function to be called at each optimization step;
            arguments are the current values of all optimization variables
            flattened into a single vector.
        :type step_callback: Callable[[np.ndarray], None]
        :param kwargs: This is a dictionary of extra parameters for session run method.
        """
        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        if model.is_built_coherence() is Build.NO:
            raise GPflowError('Model is not built.')

        if self._model is not None and self._model is not model:
            raise ValueError("Optimizer used with another model. Create new optimizer or reset existing.")

        existing_optimizer = self._optimizer is not None
        if not existing_optimizer:
            self._optimizer = self.make_optimize_tensor(model, session, var_list=var_list, maxiter=maxiter, disp=disp)

        if self._model is None or initialize:
            model.initialize(session=session)

        if self._model is None:
            self._model = model

        feed_dict = self._gen_feed_dict(model, feed_dict)
        session = model.enquire_session(session)

        if existing_optimizer and not initialize_optimizer:
            try:
                options = dict(options=dict(maxiter=maxiter, disp=disp))
                self._optimizer.optimizer_kwargs.update(options)
                self._optimizer.optimize(session=session, feed_dict=feed_dict, step_callback=step_callback)
            except InvalidArgumentError as error:
                msg = ("This error might occur because the internal state (for example, variables shape or dtype) of the model is changed. "
                       "In this case, you have to use a new optimiser")
                msg = f"Original error message: \n\t{error}\nOptimiser message: {msg}."
                raise RuntimeError(msg)
            except Exception as error:
                msg = "Unknown error has occured at reusage of the scipy optimizer. Make sure that you use the same session"
                msg = f"Original error message: \n\t{error}\nOptimiser message: {msg}."
                raise RuntimeError(msg)
        else:
            self._optimizer.minimize(session=session, feed_dict=feed_dict, step_callback=step_callback, **kwargs)

        if anchor:
            model.anchor(session)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer
