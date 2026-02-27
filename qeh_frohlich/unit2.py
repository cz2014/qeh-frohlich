"""
Physical constants and unit conversion factors for computational physics.

This module provides fundamental physical constants and conversion factors used throughout
the project for consistent unit handling. Units primarily follow Rydberg atomic units with
conversions to SI, CGS, and other common units.

Key conversions:
- Energy: Rydberg <-> eV <-> meV <-> Hartree
- Length: Bohr <-> Angstrom <-> meters
- Frequency: eV <-> s^-1 <-> THz <-> cm^-1
- Mass: atomic mass units (amu) and electron mass
"""

ry2mev = 13605.662285137
mev2ry = 1./ry2mev 

ry2ev = ry2mev*0.001
ev2ry = 1./ry2ev 

hbar = 1.054571628e-34
e_charge = 1.602176487e-19
amu = 1.660539066e-27
amuoverme = 1822.8884853323
me = amu/amuoverme
amu_ryd = amuoverme / 2.


inv_cm2mev = 0.12398
inv_cm2Thz = 0.0299793
Na = 6.02214179e23
e = 1.602176487e-19
Bohr2A = 0.529177211
Ryd2eV = 13.60569301
Ha2eV = 2 * Ryd2eV

ev2inv_s = e_charge *2 /hbar  # 3.038535163e15
inv_s2ev = 1./ev2inv_s 

evinv_a2cminv_s = 1.519267582e7

A2m = 1e-10 
A2cm = 1e-8

Ma = 1.661e-27 # 1/12 C atom mass 
Ma_cz = (95.95 + 2*32.065) * Ma  # MoS2 here 

ev2su = e_charge
mev2su = e_charge * 1e-3 

mev2omega = mev2su / hbar # equation E = \hbar \omega , value: 1.519267581699e+12

Area_cz = (3.185089*A2m)**2 * (3**0.5/2)

epsilon0 = 8.854187817e-12 

# print((hbar/2/Ma_cz/10/mev2omega)**0.5)
# print(e_charge**2/2/Area_cz/epsilon0)
# print(e_charge**2/2/Area_cz/epsilon0*(hbar/2/Ma_cz/10/mev2omega)**0.5)

C_su = e_charge**2/2/Area_cz/epsilon0*(hbar/2/Ma_cz/10/mev2omega)**0.5

if __name__ == "__main__":
    # print(e_charge*(Bohr2A*1e-8)**2/hbar)
    print((hbar**2*1e20/e/amu))
    print((e**2/hbar**2) / (e*1e20/amu))