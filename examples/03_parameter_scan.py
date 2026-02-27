"""Sensitivity analysis: Born charge scaling vs force constant softening.

Uses 3L BN + InSe + 3L BN encapsulation.
NOTE: Each (BCx, FCpow) pair takes ~2 seconds. Full 5x5 scan: ~50 seconds.
"""
import numpy as np
import matplotlib.pyplot as plt
from qeh_frohlich import FrohlichHeterostructure, find_data_file
from qeh_frohlich.qe_io import dyn_qe2qeh

BCx_values = [0.5, 0.75, 1.0, 1.25, 1.5]
FCpow_values = [0, 2, 4, 6, 8]

results = np.zeros((len(BCx_values), len(FCpow_values)))

for i, BCx in enumerate(BCx_values):
    for j, FCpow in enumerate(FCpow_values):
        # Regenerate BN phonons with scaled parameters
        dyn_qe2qeh(
            find_data_file('dynfiles/bn.dyn1'),
            'BN-phonons.npz',
            restart=False, BCx=BCx, FCpow=FCpow,
        )
        layers = 3 * ['BN+froh'] + ['InSe+froh'] + 3 * ['BN+froh']
        calc = FrohlichHeterostructure(
            layers=layers,
            effmass={'InSe': 0.192},
            mobility_layers=[3],  # InSe is layer 3 in this sandwich
        )
        mob = calc.get_mobility()
        results[i, j] = mob[0]
        print(f"BCx={BCx:.2f}, FCpow={FCpow}: {mob[0]:.1f} cm^2/Vs")

fig, ax = plt.subplots()
im = ax.imshow(results, origin='lower', aspect='auto')
ax.set_xticks(range(len(FCpow_values)))
ax.set_xticklabels(FCpow_values)
ax.set_yticks(range(len(BCx_values)))
ax.set_yticklabels(BCx_values)
ax.set_xlabel('FCpow (force constant softening)')
ax.set_ylabel('BCx (Born charge scaling)')
ax.set_title('Mobility Landscape (cm$^2$/Vs)')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('parameter_scan.png', dpi=150)
plt.show()
