from __future__ import annotations

import pytest

from innvestigate.analyzer import PatternAttribution
from innvestigate.analyzer import PatternNet

from tests.pytest_utils import dryrun

# TODO: add again a train/test case for mnist


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__PatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "imagenet.vgg16:imagenet.vgg19")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternAttribution():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternAttribution():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__PatternAttribution():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "imagenet.vgg16:imagenet.vgg19")


###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializePatternNet():
    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights() if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_serialize_analyzer(method, "mnist.log_reg")
