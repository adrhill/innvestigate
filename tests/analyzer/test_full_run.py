# from __future__ import annotations

# import os

# import pytest

# from innvestigate.analyzer import LRPEpsilon

# from tests import dryrun

# ROOT_DIR = os.path.abspath(os.curdir)

# LRP_ANALYZERS = [
#     "LRPZ",
#     "LRPZIgnoreBias",
#     "LRPEpsilon",
#     "LRPEpsilonIgnoreBias",
#     "LRPWSquare",
#     "LRPFlat",
#     "LRPAlpha2Beta1",
#     "LRPAlpha2Beta1IgnoreBias",
#     "LRPAlpha1Beta0",
#     "LRPAlpha1Beta0IgnoreBias",
#     "LRPZPlus",
#     "LRPZPlusFast",
#     "LRPSequentialPresetA",
#     "LRPSequentialPresetB",
#     "LRPSequentialPresetAFlat",
#     "LRPSequentialPresetBFlat",
#     "LRPSequentialPresetBFlatUntilIdx",
# ]

# GRADIENT_BASED_ANALYZERS = [
#     "Gradient",
#     "BaselineGradient",
#     # "InputTimesGradient",
#     "Deconvnet",
#     "GuidedBackprop",
#     "IntegratedGradients",
#     "SmoothGrad",
# ]

# DEEP_TAYLOR_ANALYZERS = [
#     "DeepTaylor",
#     "BoundedDeepTaylor",
# ]

# PATTERN_BASED_ANALYZERS = [
#     "PatternNet",
#     "PatternAttribution",
# ]


# ANALYZERS = (
#     LRP_ANALYZERS
#     + GRADIENT_BASED_ANALYZERS
#     + DEEP_TAYLOR_ANALYZERS
#     + PATTERN_BASED_ANALYZERS
# )


# @pytest.mark.slow
# def test_full_BaselineGradient():
#     def method(model):
#         return BaselineGradient(model)

#     dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


# def download_test_data() -> None:
#     """Downloads test data from test-data-innvestigate GitHub repo."""
#     url = "https://github.com/adrhill/test-data-innvestigate"
