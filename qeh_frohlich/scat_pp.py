"""
Phonon scattering rate and carrier mobility calculations.

This module calculates phonon scattering rates and carrier mobility using Frohlich
potentials from heterostructure calculations. Implements Fermi's golden rule and
relaxation time approximation to predict measurable transport properties.

Key classes:
- valley: Helper class for band structure in reciprocal space
  - Handles hexagonal lattices and k-space grids
  - Parabolic band approximation for carrier dispersion
  - Generates k-space circles for scattering integral evaluation

- scat: Main scattering calculation class
  - get_scat(): Calculates scattering rate at single energy
  - get_scatl(): Scattering rates over energy range
  - get_mob(): Computes mobility in cm^2/Vs
    - Integrates scattering rates weighted by Fermi-Dirac derivative
    - Uses relaxation time approximation
  - get_froh(): Interpolates Frohlich potentials on momentum grids

Helper functions:
- reader_scattering(): Reads EPW (Electron-Phonon Wannier) output
- reader_gepw(): Reads electron-phonon matrix elements
- reader_selfen(): Reads self-energy data

Final output: phonon-limited carrier mobility for comparison with experiments.
"""

from matplotlib.pyplot import plot
from scipy.interpolate.interpolate import interp1d
from . import unit2
import cmath
import numpy as np

class valley():
    Bohr = unit2.Bohr2A
    Hatree = unit2.Ha2eV

    def rc2pc(self, x, y):
        return cmath.polar(complex(x,y))
    
    def pc2rc(self, r, a):
        cn1 = cmath.rect(r, a)
        return cn1.real, cn1.imag

    def para_E2k(self, E): # in eV
        return (2*self.effmass*E/self.Hatree)**0.5
    def para_k2E(self, k): # in Bohr
        return k**2/2/self.effmass * self.Hatree

    def print_k(self, k, outmode='tpi'):
        """
        tpi: 2\pi/Bohr
        ang: A-1
        tpiba: 2\pi/a
        frac: fractional
        """
        if 'tpiba' in outmode:
            k = k * self.a
        elif 'ang' in outmode:
            k = k*2*np.pi / self.Bohr
        elif 'frac' in outmode:
            k = np.dot(k, self.reci_inv)
        print("%20.12f %20.12f %20.12f   1" %(k[0], k[1], 0.0))


    def __init__(self, ibrav='hex', effmass=1.0, bm=[0,0], a=1):

        ########### lattice vector
        self.a = a # in Bohr
        if isinstance(ibrav, int):
            if ibrav == 4:
                self.reci_vec = np.array([[1.000000, 0.577350],
                                        [0.000000, 1.154701]])
        elif isinstance(ibrav, str):
            if 'hex' in ibrav:
                self.reci_vec = np.array([[1.000000, 0.577350],
                                        [0.000000, 1.154701]])
        else:
            self.reci_vec = np.array(ibrav)
        self.reci_vec = self.reci_vec / self.a # in 2\pi/Bohr
        self.reci_inv = np.linalg.inv(self.reci_vec) # in Bohr
        
        ########### band minimum/maximum
        self.bm = np.array(bm)

        ########### effective mass
        self.effmass = effmass 

    def get_circle(self, num=12, initE=0.05, initA=0.0, outmode='tpi'):

        spk_r = self.para_E2k(initE) /2/np.pi # in 2\pi/Bohr

        origin_spk = np.dot(self.bm, self.reci_vec)
        nspks = num + 1

        spq_circ = np.zeros((nspks, 2))
        for i in range(nspks):
            at = initA + np.pi*i/(nspks-1)
            spq_circ[i] = self.pc2rc(spk_r, at)
        
        spq_circ = spq_circ + origin_spk[np.newaxis, :]
        spq_circ[1:] = spq_circ[1:] - spq_circ[0]
        
        if outmode:
            print('init k:')
            self.print_k(spq_circ[0], outmode=outmode)
            print('q points:')
            for i in range(num):
                self.print_k(spq_circ[i+1], outmode=outmode)
        
        return spq_circ*2*np.pi / self.Bohr # in A-1


class scat():
    bohr = unit2.Bohr2A
    hartree = unit2.Ha2eV
    ryd = hartree / 2
    kTev = 0.0259
    kT = kTev / hartree
    # at2cmvs = unit2.e_charge*(unit2.Bohr2A*1e-8)**2/unit2.hbar
    at2cmvs = unit2.hbar/unit2.me /2/hartree * 10000 # 0.02127191
    ev2invps = unit2.ev2inv_s / 1e12

    def dis1(self, Ei, w): # in hartree
        Ef = Ei + w
        res = 1./(np.exp((Ef-self.fermi)/self.kT)+1) 
        res += 1./(np.exp(w/self.kT) - 1.0)
        return res
    def dis2(self, Ei, w): 
        Ef = Ei - w
        if Ef <= 0: return Ei*0
        res = 1-1./(np.exp((Ef-self.fermi)/self.kT)+1)
        res += 1./(np.exp(w/self.kT) - 1.0)
        return res 

    def sp_sqrt(self, a):
        b = np.zeros(np.shape(a))
        b[a<0] = -1*np.sqrt(-a[a<0])
        b[a>0] = np.sqrt(a[a>0])
        return b

    def set_fermi(self, fermiev):
        self.fermiev = fermiev
        self.fermi = fermiev/self.hartree

    def __init__(self, a, effmass, bbfile):
        """
        a: lattice constant, in bohr
        effmass: effective mass, in me
        bbfile: str
        """

        ########### latt para and area
        self.a = a
        self.area = a*a*3**0.5/2 #FIXME: only for hex 
        self.abz = (2*np.pi)**2 / self.area

        ########### valley
        self.effmass = effmass
        self.va = valley(ibrav='hex', effmass=effmass, a=a, bm=[1/3,1/3]) #FIXME

        ########### g matrtix and frequency
        bbdct = np.load(bbfile)
        self.qlen = bbdct['qlen'] # Bohr^-1 unit, [nq]
        self.f2q = bbdct['f2q'] # Hartree^2 unit, [nq, nmode]
        self.fwq = self.sp_sqrt(self.f2q)
        self.gm = bbdct['froh'] # Hartree unit [nq, nmode, nlayer]
        self.nq, self.nmode, self.nlayer = np.shape(self.gm)

        ########### fermi
        self.set_fermi(-0.1)

    def get_froh(self, im, il, num=120, initE=0.05, initA=0.0, plotax=None):

        sp_circ = self.va.get_circle(num=num, initE=initE, initA=initA, 
            outmode=None)
        lenq = np.linalg.norm(sp_circ[1:], axis=1) # in A^-1

        gml = np.abs(np.real(self.gm[:,im,il]))
        gfun = interp1d(self.qlen, gml, bounds_error=False,
            fill_value=(gml[0], gml[-1]), kind='linear')

        gfroh = gfun(lenq*self.bohr) * self.hartree *1000
        gfroh = np.abs(gfroh)


        if plotax:
            plotax.plot(lenq, gfroh, label="model (im,il):"+str((im,il)))

        return lenq, gfroh

    def get_scat(self, initE, fermi=None, intn=1000, useppw=True, serta=False, splayer=None, freq_min=0.01, freq_max=None):
        if fermi is not None: self.set_fermi(fermi)
        if splayer is None: splayer = list(range(self.nlayer))
        Ei = initE / self.hartree
        k0 = self.va.para_E2k(initE)

        const = 4*np.pi*self.effmass/self.abz

        costh = np.cos(np.linspace(0, np.pi, intn))
        if serta: costh[:] = 0.0
        dtheta = np.pi/intn
        scatt = np.zeros((len(splayer),))
        for im in range(self.nmode):
            ##### FIXME: which LO freq used
            if useppw:
                dk = 2*self.va.para_E2k(initE) # in Bohr^-1
                # print(initE, " eV:", dk/unit2.Bohr2A, " A^-1")
                w = self.fwq[np.argmin(np.abs(self.qlen-dk)), im]
            else:
                w = self.fwq[0, im]
            w_ev = w * self.hartree
            if w_ev < freq_min or (freq_max is not None and w_ev > freq_max): continue
            # if im!=13 : continue # FIXME: for test

            fn1 = self.dis1(Ei, w)
            fn2 = self.dis2(Ei, w)

            for ii, il in enumerate(splayer):
                gml = np.abs(self.gm[:,im,il])
                gfun = interp1d(self.qlen, gml, bounds_error=False, 
                    fill_value=(gml[0], gml[-1]), kind='linear')
                
                ## first part
                k1 = self.va.para_E2k(initE+w_ev)
                q1 = (k0**2+k1**2-2*k0*k1*costh)**0.5
                scatt[ii] += np.dot(gfun(q1)**2, 1-costh)*fn1*dtheta

                ## second part
                if initE < w_ev: continue
                k2 = self.va.para_E2k(initE-w_ev)
                q2 = (k0**2+k2**2-2*k0*k2*costh)**0.5
                scatt[ii] += np.dot(gfun(q2)**2, 1-costh)*fn2*dtheta

        scatt = scatt*const

        return scatt # in Hartree

    def get_scatl(self, initEl, fermi=None, intn=1000, splayer=None, freq_min=0.01, freq_max=None):
        """
        initEl: energy in eV, [nE]
        return: scattering rates in eV, [nE, nlayer]
        """

        scats = np.array([self.get_scat(ie, fermi, intn, splayer=splayer, freq_min=freq_min, freq_max=freq_max) for ie in initEl])
        self.scats = scats # in Hartree unit

        return scats*self.hartree#*2**0.5 #FIXME: mysterious factor here

    def get_mob(self, fermi=None, fth=0.3, eintn=1000, sintn=100, plotax=None, splayer=None, freq_min=0.01, freq_max=None):
        """
        fth: fermi thickness in eV
        """

        if fermi is not None: self.set_fermi(fermi)

        def fd(el): # in hartree
            return 1./(np.exp((el-self.fermi)/self.kT)+1)
        def pfd(el):
            exp = np.exp((self.fermi-el)/self.kT)
            return exp/(1+exp)**2/self.kT


        enev = np.linspace(max(1e-8, self.fermiev-fth), self.fermiev+fth,
            eintn, endpoint=True)
        en = enev / self.hartree
        de = (en[-1]-en[0]) / eintn
        sc = self.get_scatl(enev, fermi=None, intn=sintn, splayer=splayer, freq_min=freq_min, freq_max=freq_max) # in ev
        tau = 1./sc*self.hartree

        if plotax:
            for i in range(len(sc[0])):
                plotax.plot(enev, sc[:,i], label="layer: "+str(i))
            plotax.set_ylabel("scat (1/ps)")
            plotax.set_xlabel("E (eV)")
            plotax.plot(enev, en*pfd(en))

        mob = np.sum((en*pfd(en))[:,np.newaxis]
            *tau*de, axis=0) / np.sum(fd(en)*de)

        mob = mob *2 /self.effmass

        return mob * self.at2cmvs
    
    def get_freq(self):
        """ get frequency in meV"""
        q = self.qlen / self.bohr
        freq = self.fwq * self.hartree * 1000

        return q, freq


def reader_scattering(fn="scattering_rate_300.00", plotax=None):
    """
    read scattering rates generated by EPW transport funcion
    """
    with open(fn, 'r') as fo:
        lines = fo.readlines()

    el = np.array([float(l.split()[2]) for l in lines[2:]])
    sc = np.array([float(l.split()[3]) for l in lines[2:]])

    el = el - np.min(el)
    if plotax:
        plotax.scatter(el, sc / (unit2.ev2inv_s/2/1e12), label="EPW")

    return (el, sc / (unit2.ev2inv_s/2/1e12))

def reader_gepw(fn="epw_lr.dat", plotax=None, lenq=None):
    with open(fn,'r') as fo:
        lines = fo.readlines()
    g = np.array([[float(entry) for entry in l.split()] for l in lines])

    if plotax:
        for i in range(len(g[0])):
            plotax.plot(lenq, g[:,i], label="EPW: m"+str(i))

    return lenq, g

def reader_selfen(filename='scatfiles/l_eps0.dat', plotax=None):
    with open(filename, 'r') as fo:
        lines = fo.readlines()

    nq = int(lines[-1].split()[0])
    scat = np.zeros((nq,2))
    for l in lines:
        if (l and "#" not in l):
            iq = int(l.split()[0]) - 1
            scat[iq,0] = float(l.split()[2])
            scat[iq,1] += float(l.split()[4])/1000
            
    scat[:,0] = scat[:,0] - np.min(scat[:,0])
    scat[:,1] = scat[:,1]*2

    if plotax:
        plotax.scatter(scat[:,0], scat[:,1], label="EPW")

    return scat


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    inse = {'a': 7.7271278, 'effmass': 0.192, 'bbfile': 'bbdct/inse_sp.npz'}
    inse = {'a': 7.7271278, 'effmass': 0.192, 'bbfile': 'bbdct/inse-sp0-all_is.npz'}
    # inse = {'a': 6.02, 'effmass': 0.45, 'bbfile': 'bbdct/inse-sp0-all_is.npz'} # mos2

    NMAT=1
    inseind = [7]*NMAT
    inseind = [0]*NMAT
    # inseind = [0,1,2,3,4,5]
    # inseind = [7,10]

    for i in range(NMAT):
        fig, ax = plt.subplots()

        inse_all_is = {'a': 7.7271278, 'effmass': 0.192, 
                'bbfile': 'bbdct/inse-sp'+str(i)+'-all_is.npz'}

        inse_ph_is = {'a': 7.7271278, 'effmass': 0.192, 
                'bbfile': 'bbdct/inse-sp'+str(i)+'-ph_is.npz'}

        inse_cp = {'a': 7.7271278, 'effmass': 0.192, 
                'bbfile': 'bbdct/inse-sp'+str(i)+'-cp.npz'}

        # matl = [inse_all_is, inse_ph_is, inse_cp]
        # matl = [inse_cp]
        matl = [inse]

        Elist = np.linspace(1e-8, 0.2, num=100)
        for imat, mat in enumerate(matl):
            mat_scat = scat(**mat)
            # print(mat_scat.va.para_E2k(0.18)/0.529177)
            # print(mat_scat.va.para_E2k(0.02)/0.529177)
            scatl = mat_scat.get_scatl(Elist, fermi=-0.1, intn=1000)
            # for il in range(mat_scat.nlayer):
            for il in [inseind[i]]:
                lb = str(il)+'-th layer; '+str(mat['bbfile'])+'-th mat'
                ax.plot(Elist, scatl[:,il], label=lb)

        # plt.legend()
        # plt.show()
        # plt.close()


        for imat, mat in enumerate(matl):
            mat_scat = scat(**mat)
            print(mat_scat.get_mob(fermi=-0.1, eintn=1000, sintn=300, plotax=False, splayer=[0]))
            # print(mat_scat.get_mob(fermi=-0.1, eintn=1000, sintn=300, plotax=ax))
            # mat_scat.va.get_circle(num=120, initE=0.05, initA=0.0, outmode='frac')
            # lenq, _ = mat_scat.get_froh(6,0,num=120, initE=0.05, plotax=ax)

        # reader_scattering("scatfiles/inse_sm10.dat", plotax=ax)
        # reader_scattering("scatfiles/inse_sm05.dat", plotax=ax)
        # reader_scattering("scatfiles/inse_sm02.dat", plotax=ax)
        # reader_scattering("scatfiles/scat_lr.dat", plotax=ax)
        reader_scattering("scatfiles/scat_lr_k600.dat", plotax=ax) #best lr res
        reader_selfen("scatfiles/l_eps0.dat", plotax=ax) # used in draft



        # reader_gepw("scatfiles/epw_ac_lr.dat", plotax=ax, lenq=lenq)

        # plt.xlim(0,0.15)
        plt.xlim(0,0.2)
        # plt.ylim(0,0.2)
        plt.legend()
        plt.show()
        plt.close()

    