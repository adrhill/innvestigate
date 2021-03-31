"""Test function 'innvestigate.create_analyzer'"""
from __future__ import annotations

import logging

import keras.layers
import keras.models
import pytest

from innvestigate import create_analyzer
from innvestigate.analyzer import analyzers


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers():

    fake_model = keras.models.Sequential([keras.layers.Dense(10, input_shape=(10,))])
    for name in analyzers.keys():
        try:
            create_analyzer(name, fake_model)
        except KeyError:
            # Name should be found!
            raise
        except:  # TODO: fix bare except
            # Some analyzers require parameters...
            pass


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__create_analyzers_wrong_name():

    fake_model = keras.models.Sequential([keras.layers.Dense(10, input_shape=(10,))])
    with pytest.raises(KeyError):
        create_analyzer("wrong name", fake_model)
