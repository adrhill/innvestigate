from __future__ import annotations

import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate.utils as iutils

__all__ = [
    "TestAnalysisHelper",
]


class TestAnalysisHelper(object):
    def __init__(self, model, analyzer, weights=None):
        """Helper class for retrieving output and analysis in test cases.


        :param model: A Keras layer object or a list of layer objects.
          In this case a sequntial model will be build. The first layer
          must have set input_shape or batch_input_shape.
          Alternatively a tuple with input and output tensors, in which
          case the keras modle api will be used.
        :param analyzer: Either an analyzer class or a function
          that takes a keras model and returns an analyzer.
        :param weights: After creating the model set the given weights.
        """

        if isinstance(model, klayers.Layer):
            model = [model]

        if isinstance(model, list):
            self._model = kmodels.Sequential(model)
        else:
            self._model = kmodels.Model(*model)

        self._input_shapes = iutils.to_list(self._model.input_shape)

        if weights is not None:
            self._model.set_weights(weights)

        self._analyzer = analyzer(self._model)

    @property
    def weights(self):
        return self._model.get_weights()

    def run(self, inputs):
        """Runs the model given the inputs.

        :return: Tuple with model output and analyzer output.
        """
        return_list = True
        if not isinstance(inputs, list):
            return_list = False
            inputs = iutils.to_list(inputs)

        augmented = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == len(self._input_shapes[i]) - 1:
                # Augment by batch axis.
                augmented.append(i)
                inputs[i] = inp.reshape((1,) + inp.shape)

        outputs = iutils.to_list(self._model.predict_on_batch(inputs))
        analysis = iutils.to_list(self._analyzer.analyze(inputs))

        for i in augmented:
            # Remove batch axis.
            outputs[i] = outputs[i][0]
            analysis[i] = analysis[i][0]

        if return_list:
            return outputs, analysis
        else:
            return outputs[0], analysis[0]
