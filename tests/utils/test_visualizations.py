from __future__ import absolute_import, division, print_function, unicode_literals

import innvestigate.utils.visualizations as ivis

import numpy as np

import pytest


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__visualizations():
    def get_X():
        return np.random.rand(1, 28, 28, 3)

    ivis.project(get_X())
    ivis.heatmap(get_X())
    ivis.graymap(get_X())
    ivis.gamma(get_X())
    ivis.clip_quantile(get_X(), 0.95)
