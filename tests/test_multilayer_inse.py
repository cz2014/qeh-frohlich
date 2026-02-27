"""Regression test for qeh-frohlich package.

Validates single-layer InSe mobility = 124.789 cm^2/Vs.
This exercises the full pipeline: heterostructure -> Frohlich -> scattering -> mobility.
"""
import os
import glob
import numpy as np
import pytest
from contextlib import contextmanager
from qeh_frohlich.qeh import make_heterostructure, Hartree, Bohr
from qeh_frohlich import frohlich as fh
from qeh_frohlich.scat_pp import scat

INSE_PARAMS = {
    'a': 7.7271278,
    'effmass': 0.192,
    'fermi': -0.1,
    'eintn': 1000,
    'sintn': 300,
}

GRID_PARAMS = {
    'frequencies': [1e-8, 1e-8, 1],
    'momenta': [0.0001, 0.4, 1000],
    'thicknesses': None,
}

EXPECTED_MOBILITY = 124.789119


@contextmanager
def cleanup_npz():
    """Remove .npz files created during test."""
    existing = set(glob.glob('./*.npz'))
    try:
        yield
    finally:
        for f in set(glob.glob('./*.npz')) - existing:
            os.remove(f)


def test_single_inse_mobility():
    """Single InSe layer mobility must match baseline value."""
    with cleanup_npz():
        layers = ['InSe+froh']
        het = make_heterostructure(layers=layers, **GRID_PARAMS)

        _, _, Einv = het.get_E_matrix(exclude_self_lattice=False)
        qlen, _, _ = het.get_dielectric_function(layer=0)

        bbset = fh.BBset()
        bbset.get_frohlich(qlen, Einv, layers, isolated=False)
        bbset.save_froh('test-sp0-cp.npz')

        mat_scat = scat(
            a=INSE_PARAMS['a'],
            effmass=INSE_PARAMS['effmass'],
            bbfile='test-sp0-cp.npz',
        )
        mobilities = mat_scat.get_mob(
            fermi=INSE_PARAMS['fermi'],
            eintn=INSE_PARAMS['eintn'],
            sintn=INSE_PARAMS['sintn'],
            plotax=None,
            splayer=[0],
            freq_min=0.0,
            freq_max=1.0,
        )

        assert abs(mobilities[0] - EXPECTED_MOBILITY) < 0.01, \
            f"Mobility {mobilities[0]:.6f} != expected {EXPECTED_MOBILITY:.6f}"
