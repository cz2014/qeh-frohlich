"""Frohlich electron-phonon coupling for van der Waals heterostructures."""

import os
import numpy as np

__version__ = "0.1.0"

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def find_data_file(filename):
    """Resolve a data file: check cwd first, then package data directory."""
    local = os.path.join(os.getcwd(), filename)
    if os.path.isfile(local):
        return local
    pkg = os.path.join(DATA_DIR, filename)
    if os.path.isfile(pkg):
        return pkg
    raise FileNotFoundError(
        f"Data file '{filename}' not found in cwd ({os.getcwd()}) "
        f"or package data ({DATA_DIR})"
    )


from . import unit2
from .qeh import make_heterostructure, Heterostructure, Hartree, Bohr
from .frohlich import BBset
from .scat_pp import scat, valley
from .qe_io import dyn_qe2qeh


class FrohlichHeterostructure:
    """Frohlich electron-phonon coupling analysis for van der Waals heterostructures.

    Parameters
    ----------
    layers : list of str
        Layer specification. Use '+froh' modifier to indicate layers with
        phonon data for Frohlich coupling. Do NOT combine +froh with +phonons
        (they are mutually exclusive).
    effmass : dict, optional
        Electron effective mass per material, in units of m_e.
    lattice_constant : dict, optional
        Lattice constant per material, in Bohr.
    mobility_layers : list of int, optional
        Layer indices for which to calculate mobility (default: [0]).
    fermi : float, optional
        Fermi level in eV (default: -0.1).
    frohlich_mode : bool or str, optional
        Frohlich coupling mode: False (coupled), 'all' (isolated),
        'ph' (phonon-normalized). Default: False.
    grid_params : dict, optional
        Override grid parameters. Keys: 'frequencies', 'momenta', 'thicknesses'.
    """

    def __init__(self, layers, effmass=None, lattice_constant=None,
                 mobility_layers=None, fermi=-0.1, frohlich_mode=False,
                 grid_params=None):
        self.layers = list(layers)
        self._validate_layers()

        self.mobility_layers = mobility_layers or [0]
        self.fermi = fermi
        self.frohlich_mode = frohlich_mode
        self.grid_params = grid_params or {
            'frequencies': [1e-8, 1e-8, 1],
            'momenta': [0.0001, 0.4, 1000],
            'thicknesses': None,
        }

        self.effmass = {}
        self.lattice_constant = {}
        for idx in self.mobility_layers:
            mat = self.layers[idx].split('+')[0]
            if effmass and mat in effmass:
                self.effmass[mat] = effmass[mat]
            else:
                self.effmass[mat] = self._lookup_effmass(mat)
            if lattice_constant and mat in lattice_constant:
                self.lattice_constant[mat] = lattice_constant[mat]
            else:
                self.lattice_constant[mat] = self._extract_lattice_constant(mat)

        self.het = None
        self.qlen = None
        self.Einv = None
        self.bbset = None
        self.dispersion = None
        self.frohlich = None
        self.scattering = None
        self.mobility = None
        self._scat_objs = {}

    def _validate_layers(self):
        has_froh = any('+froh' in layer for layer in self.layers)
        if not has_froh:
            bare_names = [l.split('+')[0] for l in self.layers]
            raise ValueError(
                f"No layer has the +froh modifier. Frohlich coupling requires "
                f"phonon building block data.\n"
                f"Your layers: {self.layers}\n"
                f"Did you mean: {[n + '+froh' for n in bare_names]}?\n"
                f"The +froh modifier signals that a -phonons.npz file is "
                f"available for this material."
            )
        for layer in self.layers:
            if '+froh' in layer and '+phonons' in layer:
                raise ValueError(
                    f"Layer '{layer}' has both +froh and +phonons modifiers. "
                    f"These are mutually exclusive.\n"
                    f"+froh: phonon data for Frohlich coupling (this package)\n"
                    f"+phonons: lattice polarizability in QEH chi "
                    f"(conflicts with Frohlich model)\n"
                    f"Use +froh for Frohlich scattering calculations."
                )

    def _lookup_effmass(self, material):
        from .qeh import default_ehmasses
        for key, val in default_ehmasses.items():
            if material in key:
                return val['emass1']
        raise ValueError(
            f"No default effective mass for '{material}'. "
            f"Please provide effmass={{'{material}': <value>}}.\n"
            f"Materials with built-in defaults: "
            f"{sorted(set(k.split('-NM')[0].split('-FM')[0] for k in default_ehmasses))}"
        )

    def _extract_lattice_constant(self, material):
        phonon_file = find_data_file(material + '-phonons.npz')
        cell = np.load(phonon_file)['cell']
        return cell[0, 0] / unit2.Bohr2A

    def build(self):
        """Build the heterostructure and compute screening matrix."""
        self.het = make_heterostructure(layers=self.layers, **self.grid_params)
        _, _, self.Einv = self.het.get_E_matrix(exclude_self_lattice=False)
        self.qlen, _, _ = self.het.get_dielectric_function(layer=0)
        return self.het

    def get_dispersion(self):
        """Compute phonon dispersion in the heterostructure.

        Returns
        -------
        frequencies : ndarray, shape [nq, nmodes]
            Phonon frequencies in meV.
        qlen : ndarray, shape [nq]
            Momentum transfer in Angstrom^-1.
        """
        if self.het is None:
            self.build()

        if self.bbset is None:
            self.bbset = BBset()
            self.bbset.get_frohlich(
                self.qlen, self.Einv, self.layers, isolated=self.frohlich_mode
            )

        f2q = self.bbset.bbdct['f2q']
        frequencies = np.sqrt(np.abs(f2q)) * Hartree * 1000
        qlen_ang = self.qlen / unit2.Bohr2A

        self.dispersion = {'qlen': qlen_ang, 'frequencies': frequencies}
        return frequencies, qlen_ang

    def get_frohlich_potential(self):
        """Compute Frohlich coupling potentials V(q, nu).

        Returns
        -------
        potentials : ndarray, shape [nq, nmodes, nlayers]
            Frohlich potentials in meV.
        frequencies : ndarray, shape [nq, nmodes]
            Phonon frequencies in meV.
        """
        if self.het is None:
            self.build()

        if self.bbset is None:
            self.bbset = BBset()
            self.bbset.get_frohlich(
                self.qlen, self.Einv, self.layers, isolated=self.frohlich_mode
            )

        froh = self.bbset.bbdct['froh']
        f2q = self.bbset.bbdct['f2q']
        potentials = np.abs(froh) * Hartree * 1000
        frequencies = np.sqrt(np.abs(f2q)) * Hartree * 1000

        self.frohlich = {
            'qlen': self.qlen / unit2.Bohr2A,
            'potentials': potentials,
            'frequencies': frequencies,
        }
        return potentials, frequencies

    def get_scattering_rates(self, energies=None):
        """Compute scattering rates vs carrier energy.

        Parameters
        ----------
        energies : ndarray, optional
            Energy array in eV. Default: linspace(1e-8, 0.2, 100).

        Returns
        -------
        energies : ndarray
            Energy array (eV).
        rates : ndarray
            Scattering rates, shape [n_energies, n_layers].
        """
        if not self._scat_objs:
            self._build_scat_objs()

        if energies is None:
            energies = np.linspace(1e-8, 0.2, 100)

        all_rates = []
        for idx in self.mobility_layers:
            mat = self.layers[idx].split('+')[0]
            rates = self._scat_objs[mat].get_scatl(
                energies, fermi=self.fermi, intn=1000,
                splayer=[idx],
            )
            all_rates.append(rates)
        rates = np.column_stack(all_rates)

        self.scattering = {'energy': energies, 'rates': rates}
        return energies, rates

    def get_mobility(self):
        """Compute phonon-limited carrier mobility.

        Returns
        -------
        mobility : list of float
            Mobility in cm^2/Vs, one value per mobility_layer.
        """
        if not self._scat_objs:
            self._build_scat_objs()

        results = []
        for idx in self.mobility_layers:
            mat = self.layers[idx].split('+')[0]
            mob = self._scat_objs[mat].get_mob(
                fermi=self.fermi,
                eintn=1000, sintn=300,
                plotax=None,
                splayer=[idx],
                freq_min=0.01,
            )
            results.append(mob[0])

        self.mobility = results
        return self.mobility

    def _build_scat_objs(self):
        import tempfile

        if self.het is None:
            self.build()
        if self.bbset is None:
            self.bbset = BBset()
            self.bbset.get_frohlich(
                self.qlen, self.Einv, self.layers, isolated=self.frohlich_mode
            )

        tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        tmp_path = tmp.name
        tmp.close()
        self.bbset.save_froh(tmp_path)
        try:
            for idx in self.mobility_layers:
                mat = self.layers[idx].split('+')[0]
                if mat not in self._scat_objs:
                    self._scat_objs[mat] = scat(
                        a=self.lattice_constant[mat],
                        effmass=self.effmass[mat],
                        bbfile=tmp_path,
                    )
        finally:
            os.unlink(tmp_path)

    def run(self):
        """Execute the full pipeline: build + all physical quantities.

        Returns
        -------
        mobility : list of float
            Mobility in cm^2/Vs.
        """
        self.build()
        self.get_dispersion()
        self.get_frohlich_potential()
        self.get_scattering_rates()
        return self.get_mobility()
