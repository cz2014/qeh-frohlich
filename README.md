# qeh-frohlich

Frohlich electron-phonon coupling and phonon-limited carrier mobility in van der Waals heterostructures.

## Installation

```bash
git clone <repo-url>
cd qeh-frohlich
pip install -e .
```

## Usage

```python
from qeh_frohlich import make_heterostructure, BBset, scat

het = make_heterostructure(['InSe+lattpol'])
qlen, _, Einv = het.get_dielectric_function(layer=0)
_, _, Einv = het.get_E_matrix()

bbset = BBset()
bbset.get_frohlich(qlen, Einv, ['InSe+lattpol'], isolated=False)
bbset.save_froh('froh.npz')

s = scat(a=7.727, effmass=0.192, bbfile='froh.npz')
print(s.get_mob(fermi=-0.1, eintn=1000, sintn=300))
```

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).
