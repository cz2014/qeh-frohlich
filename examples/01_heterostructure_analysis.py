"""In-depth analysis of a BN-encapsulated InSe heterostructure.

Demonstrates all physical quantities obtainable from the package:
1. Phonon dispersion in the heterostructure
2. Frohlich electron-phonon coupling potentials
3. Energy-resolved scattering rates
4. Carrier mobility
5. Comparison of coupled vs isolated Frohlich potentials
"""
import numpy as np
import matplotlib.pyplot as plt
from qeh_frohlich import FrohlichHeterostructure, BBset, Hartree

# --- Setup: InSe encapsulated by 5 BN layers on each side ---
layers = 5 * ['BN+froh'] + ['InSe+froh'] + 5 * ['BN+froh']
calc = FrohlichHeterostructure(
    layers=layers,
    effmass={'InSe': 0.192},
    mobility_layers=[5],        # InSe is layer 5
)

# --- Run full pipeline ---
calc.run()
print(f"Mobility: {calc.mobility[0]:.3f} cm^2/Vs")

# --- Plot 1: Phonon dispersion ---
freqs, qlen = calc.get_dispersion()
potentials, _ = calc.get_frohlich_potential()
THRESHOLD = 9.0  # meV

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Dispersion with highlighted modes
nmode = freqs.shape[1]
froh_layer5 = potentials[:, :, 5]  # Frohlich at InSe layer

for mode in range(nmode):
    axes[0].plot(qlen, freqs[:, mode], color='grey', alpha=0.3, lw=1)

selected = [m for m in range(nmode) if np.max(froh_layer5[:, m]) > THRESHOLD]
colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))
for i, mode in enumerate(selected):
    axes[0].plot(qlen, freqs[:, mode], color=colors[i], lw=2, label=f'mode {mode}')

axes[0].set_xlabel(r'q ($\AA^{-1}$)')
axes[0].set_ylabel('Frequency (meV)')
axes[0].set_title('Phonon Dispersion')
axes[0].grid(True, alpha=0.3)
if selected:
    axes[0].legend(fontsize=8)

# --- Plot 2: Frohlich potentials ---
for i, mode in enumerate(selected):
    axes[1].plot(qlen, froh_layer5[:, mode], color=colors[i], lw=2,
                 label=f'mode {mode}')

axes[1].set_xlabel(r'q ($\AA^{-1}$)')
axes[1].set_ylabel('|V(q)| (meV)')
axes[1].set_title('Frohlich Potential (InSe layer)')
axes[1].grid(True, alpha=0.3)
if selected:
    axes[1].legend(fontsize=8)

# --- Plot 3: Scattering rates ---
energies, rates = calc.get_scattering_rates()
axes[2].plot(energies, rates[:, 0], lw=2)
axes[2].set_xlabel('Energy (eV)')
axes[2].set_ylabel('Scattering Rate (1/ps)')
axes[2].set_title('Phonon Scattering Rate')
axes[2].grid(True, alpha=0.3)
axes[2].annotate(
    f'Mobility: {calc.mobility[0]:.1f} cm$^2$/Vs',
    xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
    fontsize=10, bbox=dict(boxstyle='round', fc='wheat', alpha=0.5)
)

plt.tight_layout()
plt.savefig('overview.png', dpi=150)
plt.show()

# --- Compare isolated vs coupled Frohlich ---
bbset_isolated = BBset()
bbset_isolated.get_frohlich(
    calc.qlen, calc.Einv, layers, isolated='all'
)

froh_coupled = np.abs(calc.bbset.bbdct['froh']) * Hartree * 1000  # meV
froh_isolated = np.abs(bbset_isolated.bbdct['froh']) * Hartree * 1000

plt.figure()
max_mode = np.argmax(np.max(froh_coupled[:, :, 5], axis=0))
plt.plot(calc.qlen, froh_coupled[:, max_mode, 5], 'b-', lw=2, label='Coupled')
plt.plot(calc.qlen, froh_isolated[:, max_mode, 5], 'r--', lw=2, label='Isolated')
plt.xlabel(r'q ($\AA^{-1}$)')
plt.ylabel('|V(q)| (meV)')
plt.title(f'Frohlich Potential: Mode {max_mode} (InSe layer)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
