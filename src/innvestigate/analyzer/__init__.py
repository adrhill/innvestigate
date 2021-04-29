from __future__ import annotations

from innvestigate.analyzer.base import NotAnalyzeableModelException
from innvestigate.analyzer.deeptaylor import BoundedDeepTaylor
from innvestigate.analyzer.deeptaylor import DeepTaylor
from innvestigate.analyzer.gradient_based import BaselineGradient
from innvestigate.analyzer.gradient_based import Deconvnet
from innvestigate.analyzer.gradient_based import Gradient
from innvestigate.analyzer.gradient_based import GuidedBackprop
from innvestigate.analyzer.gradient_based import InputTimesGradient
from innvestigate.analyzer.gradient_based import IntegratedGradients
from innvestigate.analyzer.gradient_based import SmoothGrad
from innvestigate.analyzer.misc import Input
from innvestigate.analyzer.misc import Random
from innvestigate.analyzer.pattern_based import PatternAttribution
from innvestigate.analyzer.pattern_based import PatternNet
from innvestigate.analyzer.relevance_based.relevance_analyzer import BaselineLRPZ
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRP
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha1Beta0
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPAlpha1Beta0IgnoreBias,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha2Beta1
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPAlpha2Beta1IgnoreBias,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlphaBeta
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPEpsilon
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPEpsilonIgnoreBias,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPFlat
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetA,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetAFlat,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetB,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetBFlat,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import (
    LRPSequentialPresetBFlatUntilIdx,
)
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPWSquare
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZ
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZIgnoreBias
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZPlus
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPZPlusFast
from innvestigate.analyzer.wrapper import AugmentReduceBase
from innvestigate.analyzer.wrapper import GaussianSmoother
from innvestigate.analyzer.wrapper import PathIntegrator
from innvestigate.analyzer.wrapper import WrapperBase

# Disable pyflaks warnings:
assert NotAnalyzeableModelException
assert BaselineLRPZ
assert WrapperBase
assert AugmentReduceBase
assert GaussianSmoother
assert PathIntegrator


analyzers = {
    # Utility.
    "input": Input,
    "random": Random,
    # Gradient based
    "gradient": Gradient,
    "gradient.baseline": BaselineGradient,
    "input_t_gradient": InputTimesGradient,
    "deconvnet": Deconvnet,
    "guided_backprop": GuidedBackprop,
    "integrated_gradients": IntegratedGradients,
    "smoothgrad": SmoothGrad,
    # Relevance based
    "lrp": LRP,
    "lrp.z": LRPZ,
    "lrp.z_IB": LRPZIgnoreBias,
    "lrp.epsilon": LRPEpsilon,
    "lrp.epsilon_IB": LRPEpsilonIgnoreBias,
    "lrp.w_square": LRPWSquare,
    "lrp.flat": LRPFlat,
    "lrp.alpha_beta": LRPAlphaBeta,
    "lrp.alpha_2_beta_1": LRPAlpha2Beta1,
    "lrp.alpha_2_beta_1_IB": LRPAlpha2Beta1IgnoreBias,
    "lrp.alpha_1_beta_0": LRPAlpha1Beta0,
    "lrp.alpha_1_beta_0_IB": LRPAlpha1Beta0IgnoreBias,
    "lrp.z_plus": LRPZPlus,
    "lrp.z_plus_fast": LRPZPlusFast,
    "lrp.sequential_preset_a": LRPSequentialPresetA,
    "lrp.sequential_preset_b": LRPSequentialPresetB,
    "lrp.sequential_preset_a_flat": LRPSequentialPresetAFlat,
    "lrp.sequential_preset_b_flat": LRPSequentialPresetBFlat,
    "lrp.sequential_preset_b_flat_until_idx": LRPSequentialPresetBFlatUntilIdx,
    # Deep Taylor
    "deep_taylor": DeepTaylor,
    "deep_taylor.bounded": BoundedDeepTaylor,
    # Pattern based
    "pattern.net": PatternNet,
    "pattern.attribution": PatternAttribution,
}


def create_analyzer(name, model, **kwargs):
    """Instantiates the analyzer with the name 'name'

    This convenience function takes an analyzer name
    creates the respective analyzer.

    Alternatively analyzers can be created directly by
    instantiating the respective classes.

    :param name: Name of the analyzer.
    :param model: The model to analyze, passed to the analyzer's __init__.
    :param kwargs: Additional parameters for the analyzer's .
    :return: An instance of the chosen analyzer.
    :raise KeyError: If there is no analyzer with the passed name.
    """
    try:
        analyzer_class = analyzers[name]
    except KeyError:
        raise KeyError(
            "No analyzer with the name '%s' could be found."
            " All possible names are: %s" % (name, list(analyzers.keys()))
        )
    return analyzer_class(model, **kwargs)
