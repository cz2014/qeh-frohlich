"""Physical constants and unit conversion factors.

Units follow Rydberg atomic units with conversions to SI and common units.
"""

# Energy
ry2ev = 13.60569301
ev2ry = 1.0 / ry2ev
Ha2eV = 2 * ry2ev

# Length
Bohr2A = 0.529177211

# Mass
hbar = 1.054571628e-34          # J*s
e_charge = 1.602176487e-19      # C
amu = 1.660539066e-27           # kg
amuoverme = 1822.8884853323     # amu / electron mass
me = amu / amuoverme            # electron mass in kg
amu_ryd = amuoverme / 2.0       # amu in Rydberg mass units

# Frequency
ev2inv_s = e_charge * 2 / hbar  # eV to inverse seconds (3.038e15)
