from __future__ import annotations

import pytest

from innvestigate.analyzer import BoundedDeepTaylor
from innvestigate.analyzer import DeepTaylor

from tests.pytest_utils import dryrun


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__DeepTaylor():
    def method(model):
        return DeepTaylor(model)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__DeepTaylor():
    def method(model):
        return DeepTaylor(model)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__DeepTaylor():
    def method(model):
        return DeepTaylor(model)

    dryrun.test_analyzer(method, "imagenet.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__BoundedDeepTaylor():
    def method(model):
        return BoundedDeepTaylor(model, low=-1, high=1)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__BoundedDeepTaylor():
    def method(model):
        return BoundedDeepTaylor(model, low=-1, high=1)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.slow
@pytest.mark.application
@pytest.mark.imagenet
def test_imagenet__BoundedDeepTaylor():
    def method(model):
        return BoundedDeepTaylor(model, low=-1, high=1)

    dryrun.test_analyzer(method, "imagenet.*")
