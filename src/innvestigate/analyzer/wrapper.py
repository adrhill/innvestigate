from __future__ import annotations

import warnings
from builtins import zip
from typing import Dict, List, Union

import keras.backend
import keras.models
import numpy as np

import innvestigate.layers as ilayers
import innvestigate.utils as iutils
import innvestigate.utils.keras as kutils
from innvestigate.analyzer.base import AnalyzerBase
from innvestigate.analyzer.network_base import AnalyzerNetworkBase
from innvestigate.utils.types import Tensor

__all__ = [
    "WrapperBase",
    "AugmentReduceBase",
    "GaussianSmoother",
    "PathIntegrator",
]


class WrapperBase(AnalyzerBase):
    """Interface for wrappers around analyzers

    This class is the basic interface for wrappers around analyzers.

    :param subanalyzer: The analyzer to be wrapped.
    """

    def __init__(self, subanalyzer: AnalyzerBase, *args, **kwargs):
        # To simplify serialization, additionaly passed models are popped
        # and the subanalyzer model is passed to `AnalyzerBase`.
        kwargs.pop("model", None)
        super().__init__(subanalyzer._model, *args, **kwargs)
        self._subanalyzer = subanalyzer

    def analyze(self, *args, **kwargs):
        return self._subanalyzer.analyze(*args, **kwargs)

    def _get_state(self) -> dict:
        sa_class_name, sa_state = self._subanalyzer.save()

        state = super()._get_state()
        state.update({"subanalyzer_class_name": sa_class_name})
        state.update({"subanalyzer_state": sa_state})
        return state

    @classmethod
    def _state_to_kwargs(cls, state: dict):
        sa_class_name = state.pop("subanalyzer_class_name")
        sa_state = state.pop("subanalyzer_state")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        subanalyzer = AnalyzerBase.load(sa_class_name, sa_state)
        kwargs.update({"subanalyzer": subanalyzer})
        return kwargs


###############################################################################


class AugmentReduceBase(WrapperBase):
    """Interface for wrappers that augment the input and reduce the analysis.

    This class is an interface for wrappers that:
    * augment the input to the analyzer by creating new samples.
    * reduce the returned analysis to match the initial input shapes.

    :param subanalyzer: The analyzer to be wrapped.
    :param augment_by_n: Number of samples to create.
    """

    def __init__(
        self, subanalyzer: AnalyzerNetworkBase, *args, augment_by_n: int = 2, **kwargs
    ):
        if not isinstance(subanalyzer, AnalyzerNetworkBase):
            raise NotImplementedError("Keras-based subanalyzer required.")

        _subanalyzer_name = subanalyzer.__class__.__name__
        if subanalyzer._neuron_selection_mode == "max_activation":
            warnings.warn(
                f"Subanalyzer {_subanalyzer_name} created through AugmentReduceBase "
                f"""only supports neuron_selection_mode "all" and "index". """
                f"""Specified mode "max_activation" has been changed to "all"."""
            )
            subanalyzer._neuron_selection_mode = "all"

        if subanalyzer._neuron_selection_mode not in ["all", "index"]:
            raise NotImplementedError(
                f"Subanalyzer {_subanalyzer_name} created through AugmentReduceBase "
                f"""only supports neuron_selection_mode "all" and "index". """
                f"""got "{subanalyzer._neuron_selection_mode}"."""
            )

        super().__init__(subanalyzer, *args, **kwargs)

        self._subanalyzer = subanalyzer
        self._augment_by_n = augment_by_n
        self._neuron_selection_mode = subanalyzer._neuron_selection_mode

    def create_analyzer_model(self):
        self._subanalyzer.create_analyzer_model()

        if self._subanalyzer._n_debug_output > 0:
            raise NotImplementedError("No debug output at subanalyzer is supported.")

        model = self._subanalyzer._analyzer_model
        if None in model.input_shape[1:]:
            raise ValueError(
                "The input shape for the model needs "
                "to be fully specified (except the batch axis). "
                f"Model input shape is: {model.input_shape}"
            )

        inputs = model.inputs[: self._subanalyzer._n_data_input]
        extra_inputs = model.inputs[self._subanalyzer._n_data_input :]

        outputs = model.outputs[: self._subanalyzer._n_data_output]
        extra_outputs = model.outputs[self._subanalyzer._n_data_output :]

        if len(extra_outputs) > 0:
            raise Exception("No extra output is allowed " "with this wrapper.")

        new_inputs = iutils.to_list(self._augment(inputs))
        # print(type(new_inputs), type(extra_inputs))
        tmp = iutils.to_list(model(new_inputs + extra_inputs))
        new_outputs = iutils.to_list(self._reduce(tmp))
        new_constant_inputs = self._keras_get_constant_inputs()

        new_model = keras.models.Model(
            inputs=inputs + extra_inputs + new_constant_inputs,
            outputs=outputs + new_outputs + extra_outputs,
        )
        self._subanalyzer._analyzer_model = new_model

    def analyze(self, X: Union[np.ndarray, List[np.ndarray]], *args, **kwargs):
        if not hasattr(self._subanalyzer, "_analyzer_model"):
            self.create_analyzer_model()

        ns_mode = self._neuron_selection_mode
        # TODO: fix neuron_selection with mode "index"
        if ns_mode in ["max_activation", "index"]:
            if ns_mode == "index":
                # TODO: make neuron_selection arg or kwarg, not both
                if len(args):
                    args = list(args)
                    indices = args.pop(0)
                else:
                    indices = kwargs.pop("neuron_selection")
            # TODO: add "max_activation"
            # elif ns_mode == "max_activation":
            #     tmp = self._subanalyzer._model.predict(X)
            #     indices = np.argmax(tmp, axis=1)

            # broadcast to match augmented samples.
            indices = np.repeat(indices, self._augment_by_n)

            kwargs["neuron_selection"] = indices
        return self._subanalyzer.analyze(X, *args, **kwargs)

    def _keras_get_constant_inputs(self):
        return list()

    def _augment(self, X):
        repeat = ilayers.Repeat(self._augment_by_n, axis=0)
        return [repeat(x) for x in iutils.to_list(X)]

    def _reduce(self, X):
        X_shape = [keras.backend.int_shape(x) for x in iutils.to_list(X)]
        reshape = [
            ilayers.Reshape((-1, self._augment_by_n) + shape[1:]) for shape in X_shape
        ]
        mean = ilayers.Mean(axis=1)

        return [mean(reshape_x(x)) for x, reshape_x in zip(X, reshape)]

    def _get_state(self):
        state = super()._get_state()
        state.update({"augment_by_n": self._augment_by_n})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        augment_by_n = state.pop("augment_by_n")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"augment_by_n": augment_by_n})
        return kwargs


###############################################################################


class GaussianSmoother(AugmentReduceBase):
    """Wrapper that adds noise to the input and averages over analyses

    This wrapper creates new samples by adding Gaussian noise
    to the input. The final analysis is an average of the returned analyses.

    :param subanalyzer: The analyzer to be wrapped.
    :param noise_scale: The stddev of the applied noise.
    :param augment_by_n: Number of samples to create.
    """

    def __init__(self, subanalyzer, *args, noise_scale: float = 1, **kwargs):
        super().__init__(subanalyzer, *args, **kwargs)
        self._noise_scale = noise_scale

    def _augment(self, X):
        tmp = super()._augment(X)
        noise = ilayers.TestPhaseGaussianNoise(stddev=self._noise_scale)
        return [noise(x) for x in tmp]

    def _get_state(self):
        state = super()._get_state()
        state.update({"noise_scale": self._noise_scale})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        noise_scale = state.pop("noise_scale")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        kwargs.update({"noise_scale": noise_scale})
        return kwargs


###############################################################################


class PathIntegrator(AugmentReduceBase):
    """Integrated the analysis along a path

    This analyzer:
    * creates a path from input to reference image.
    * creates steps number of intermediate inputs and
      crests an analysis for them.
    * sums the analyses and multiplies them with the input-reference_input.

    This wrapper is used to implement Integrated Gradients.
    We refer to the paper for further information.

    :param subanalyzer: The analyzer to be wrapped.
    :param steps: Number of steps for integration.
    :param reference_inputs: The reference input.
    """

    def __init__(
        self, subanalyzer, *args, steps: int = 16, reference_inputs: int = 0, **kwargs
    ):
        super().__init__(subanalyzer, *args, augment_by_n=steps, **kwargs)

        self._reference_inputs = reference_inputs
        self._keras_constant_inputs: List[Tensor] = None

    def _keras_set_constant_inputs(self, inputs: List[Tensor]) -> None:
        tmp = [keras.backend.variable(X) for X in inputs]
        self._keras_constant_inputs = [
            keras.layers.Input(tensor=X, shape=X.shape[1:]) for X in tmp
        ]

    def _keras_get_constant_inputs(self) -> List[Tensor]:
        return self._keras_constant_inputs

    def _compute_difference(self, X: List[Tensor]) -> List[Tensor]:
        if self._keras_constant_inputs is None:
            tmp = kutils.broadcast_np_tensors_to_keras_tensors(
                X, self._reference_inputs
            )
            self._keras_set_constant_inputs(tmp)

        reference_inputs = self._keras_get_constant_inputs()
        return [keras.layers.Subtract()([x, ri]) for x, ri in zip(X, reference_inputs)]

    def _augment(self, X):
        tmp = super()._augment(X)
        tmp = [
            ilayers.Reshape((-1, self._augment_by_n) + keras.backend.int_shape(x)[1:])(
                x
            )
            for x in tmp
        ]

        difference = self._compute_difference(X)
        self._keras_difference = difference
        # Make broadcastable.
        difference = [
            ilayers.Reshape((-1, 1) + keras.backend.int_shape(x)[1:])(x)
            for x in difference
        ]

        # Compute path steps.
        multiply_with_linspace = ilayers.MultiplyWithLinspace(
            0, 1, n=self._augment_by_n, axis=1
        )
        path_steps = [multiply_with_linspace(d) for d in difference]

        reference_inputs = self._keras_get_constant_inputs()
        ret = [keras.layers.Add()([x, p]) for x, p in zip(reference_inputs, path_steps)]
        ret = [ilayers.Reshape((-1,) + keras.backend.int_shape(x)[2:])(x) for x in ret]
        return ret

    def _reduce(self, X):
        tmp = super()._reduce(X)
        difference = self._keras_difference
        del self._keras_difference

        return [keras.layers.Multiply()([x, d]) for x, d in zip(tmp, difference)]

    def _get_state(self):
        state = super()._get_state()
        state.update({"reference_inputs": self._reference_inputs})
        return state

    @classmethod
    def _state_to_kwargs(cls, state):
        reference_inputs = state.pop("reference_inputs")
        # call super after popping class-specific states:
        kwargs = super()._state_to_kwargs(state)

        # We use steps instead.
        kwargs.update(
            {"reference_inputs": reference_inputs, "steps": kwargs["augment_by_n"]}
        )
        del kwargs["augment_by_n"]
        return kwargs
