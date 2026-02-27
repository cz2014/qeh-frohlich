"""
Test for single-layer InSe mobility calculation.

Validates that the restructured package produces the same mobility
(124.789 cm^2/Vs) as the original cz_qeh3 code.
"""

import os
import glob
import numpy as np
import pytest
from contextlib import contextmanager
from qeh_frohlich.qeh import make_heterostructure, Hartree, Bohr
from qeh_frohlich import frohlich as fh
from qeh_frohlich.scat_pp import scat
from qeh_frohlich import unit2

# InSe material parameters
INSE_PARAMS = {
    'a': 7.7271278,           # Lattice constant in Bohr
    'effmass': 0.192,         # Electron effective mass in m_e
    'fermi': -0.1,            # Fermi level in eV
    'eintn': 1000,            # Energy integration points for mobility
    'sintn': 300,             # Scattering integration points
}

GRID_PARAMS = {
    'frequencies': [1e-8, 1e-8, 1],
    'momenta': [0.0001, 0.4, 1000],
    'thicknesses': None,
}

EXPECTED_MOBILITY = 124.789119  # cm^2/Vs baseline from cz_qeh3


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
        layers = ['InSe+lattpol']
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

        mobility = mobilities[0]
        print(f"Mobility: {mobility:.6f} cm^2/Vs (expected: {EXPECTED_MOBILITY:.6f})")
        assert abs(mobility - EXPECTED_MOBILITY) < 0.01, \
            f"Mobility {mobility:.6f} differs from expected {EXPECTED_MOBILITY:.6f}"


if __name__ == "__main__":
    test_single_inse_mobility()
