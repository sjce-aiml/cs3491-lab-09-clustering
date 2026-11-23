"""Public tests for Lab 09. Hidden tests are harder!"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import lab09 as lab  # noqa: E402

import numpy as np

def test_two_points_two_clusters():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels, C = lab.kmeans(X, 2, seed=0)
    assert len(set(labels)) == 2
