"""Study how BN encapsulation layers affect InSe mobility.

Sweeps from freestanding InSe (n=0) to fully encapsulated (n=10 per side).
"""
import matplotlib.pyplot as plt
from qeh_frohlich import FrohlichHeterostructure

substrate_counts = [0, 1, 2, 3, 5, 7, 10]
mobilities = []

for n in substrate_counts:
    if n == 0:
        layers = ['InSe+froh']
        mob_layers = [0]
    else:
        layers = n * ['BN+froh'] + ['InSe+froh'] + n * ['BN+froh']
        mob_layers = [n]  # InSe is at index n

    calc = FrohlichHeterostructure(
        layers=layers,
        effmass={'InSe': 0.192},
        mobility_layers=mob_layers,
    )
    mob = calc.get_mobility()
    mobilities.append(mob[0])
    print(f"N_BN={n:2d} (per side): mobility = {mob[0]:.3f} cm^2/Vs")

plt.plot(substrate_counts, mobilities, 'o-')
plt.xlabel('Number of BN layers (per side)')
plt.ylabel('Mobility (cm$^2$/Vs)')
plt.title('Remote Phonon Effect on InSe Mobility')
plt.tight_layout()
plt.savefig('substrate_sweep.png', dpi=150)
plt.show()
