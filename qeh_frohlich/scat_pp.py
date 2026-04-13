"""Phonon scattering rate and carrier mobility calculations."""

from scipy.interpolate import interp1d
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

    def para_E2k(self, E):
        return (2*self.effmass*E/self.Hatree)**0.5
    def para_k2E(self, k):
        return k**2/2/self.effmass * self.Hatree

    def print_k(self, k, outmode='tpi'):
        if 'tpiba' in outmode:
            k = k * self.a
        elif 'ang' in outmode:
            k = k*2*np.pi / self.Bohr
        elif 'frac' in outmode:
            k = np.dot(k, self.reci_inv)
        print("%20.12f %20.12f %20.12f   1" %(k[0], k[1], 0.0))

    def __init__(self, ibrav='hex', effmass=1.0, bm=[0,0], a=1):
        self.a = a
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
        self.reci_vec = self.reci_vec / self.a
        self.reci_inv = np.linalg.inv(self.reci_vec)
        self.bm = np.array(bm)
        self.effmass = effmass

    def get_circle(self, num=12, initE=0.05, initA=0.0, outmode='tpi'):
        spk_r = self.para_E2k(initE) /2/np.pi

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

        return spq_circ*2*np.pi / self.Bohr


class scat():
    bohr = unit2.Bohr2A
    hartree = unit2.Ha2eV
    ryd = hartree / 2
    kTev = 0.0259
    kT = kTev / hartree
    at2cmvs = unit2.hbar/unit2.me /2/hartree * 10000
    ev2invps = unit2.ev2inv_s / 1e12

    def dis1(self, Ei, w):
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
        self.a = a
        self.area = a*a*3**0.5/2
        self.abz = (2*np.pi)**2 / self.area

        self.effmass = effmass
        self.va = valley(ibrav='hex', effmass=effmass, a=a, bm=[1/3,1/3])

        bbdct = np.load(bbfile)
        self.qlen = bbdct['qlen']
        self.f2q = bbdct['f2q']
        self.fwq = self.sp_sqrt(self.f2q)
        self.gm = bbdct['froh']
        self.nq, self.nmode, self.nlayer = np.shape(self.gm)

        self.set_fermi(-0.1)

    def get_froh(self, im, il, num=120, initE=0.05, initA=0.0, plotax=None):
        sp_circ = self.va.get_circle(num=num, initE=initE, initA=initA,
            outmode=None)
        lenq = np.linalg.norm(sp_circ[1:], axis=1)

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
            if useppw:
                dk = 2*self.va.para_E2k(initE)
                w = self.fwq[np.argmin(np.abs(self.qlen-dk)), im]
            else:
                w = self.fwq[0, im]
            w_ev = w * self.hartree
            if w_ev < freq_min or (freq_max is not None and w_ev > freq_max): continue

            fn1 = self.dis1(Ei, w)
            fn2 = self.dis2(Ei, w)

            for ii, il in enumerate(splayer):
                gml = np.abs(self.gm[:,im,il])
                gfun = interp1d(self.qlen, gml, bounds_error=False,
                    fill_value=(gml[0], gml[-1]), kind='linear')

                k1 = self.va.para_E2k(initE+w_ev)
                q1 = (k0**2+k1**2-2*k0*k1*costh)**0.5
                scatt[ii] += np.dot(gfun(q1)**2, 1-costh)*fn1*dtheta

                if initE < w_ev: continue
                k2 = self.va.para_E2k(initE-w_ev)
                q2 = (k0**2+k2**2-2*k0*k2*costh)**0.5
                scatt[ii] += np.dot(gfun(q2)**2, 1-costh)*fn2*dtheta

        scatt = scatt*const

        return scatt

    def get_scatl(self, initEl, fermi=None, intn=1000, splayer=None, freq_min=0.01, freq_max=None):
        scats = np.array([self.get_scat(ie, fermi, intn, splayer=splayer, freq_min=freq_min, freq_max=freq_max) for ie in initEl])
        self.scats = scats

        return scats*self.hartree

    def get_scat_mr(self, initE, fermi=None, intn=1000, useppw=True,
                    serta=False, splayer=None, freq_min=0.01, freq_max=None):
        """Mode-resolved scattering rate at a single energy.

        Returns shape [nmode, len(splayer)] instead of [len(splayer)].
        """
        if fermi is not None: self.set_fermi(fermi)
        if splayer is None: splayer = list(range(self.nlayer))
        Ei = initE / self.hartree
        k0 = self.va.para_E2k(initE)

        const = 4*np.pi*self.effmass/self.abz

        costh = np.cos(np.linspace(0, np.pi, intn))
        if serta: costh[:] = 0.0
        dtheta = np.pi/intn
        scatt = np.zeros((self.nmode, len(splayer)))
        for im in range(self.nmode):
            if useppw:
                dk = 2*self.va.para_E2k(initE)
                w = self.fwq[np.argmin(np.abs(self.qlen-dk)), im]
            else:
                w = self.fwq[0, im]
            w_ev = w * self.hartree
            if w_ev < freq_min or (freq_max is not None and w_ev > freq_max): continue

            fn1 = self.dis1(Ei, w)
            fn2 = self.dis2(Ei, w)

            for ii, il in enumerate(splayer):
                gml = np.abs(self.gm[:,im,il])
                gfun = interp1d(self.qlen, gml, bounds_error=False,
                    fill_value=(gml[0], gml[-1]), kind='linear')

                k1 = self.va.para_E2k(initE+w_ev)
                q1 = (k0**2+k1**2-2*k0*k1*costh)**0.5
                scatt[im, ii] += np.dot(gfun(q1)**2, 1-costh)*fn1*dtheta

                if initE < w_ev: continue
                k2 = self.va.para_E2k(initE-w_ev)
                q2 = (k0**2+k2**2-2*k0*k2*costh)**0.5
                scatt[im, ii] += np.dot(gfun(q2)**2, 1-costh)*fn2*dtheta

        scatt = scatt*const
        return scatt

    def get_scatl_mr(self, initEl, fermi=None, intn=1000, splayer=None,
                     freq_min=0.01, freq_max=None):
        """Mode-resolved scattering rates over an energy list.

        Returns shape [nE, nmode, len(splayer)].
        """
        scats = np.array([
            self.get_scat_mr(ie, fermi, intn, splayer=splayer,
                             freq_min=freq_min, freq_max=freq_max)
            for ie in initEl
        ])
        return scats * self.hartree

    def get_mob(self, fermi=None, fth=0.3, eintn=1000, sintn=100, plotax=None, splayer=None, freq_min=0.01, freq_max=None):
        if fermi is not None: self.set_fermi(fermi)

        def fd(el):
            return 1./(np.exp((el-self.fermi)/self.kT)+1)
        def pfd(el):
            exp = np.exp((self.fermi-el)/self.kT)
            return exp/(1+exp)**2/self.kT

        enev = np.linspace(max(1e-8, self.fermiev-fth), self.fermiev+fth,
            eintn, endpoint=True)
        en = enev / self.hartree
        de = (en[-1]-en[0]) / eintn
        sc = self.get_scatl(enev, fermi=None, intn=sintn, splayer=splayer, freq_min=freq_min, freq_max=freq_max)
        tau = 1./sc*self.hartree

        if plotax:
            for i in range(len(sc[0])):
                plotax.plot(enev, sc[:,i], label="layer: "+str(i))
            plotax.set_ylabel("scat (eV)")
            plotax.set_xlabel("E (eV)")
            plotax.plot(enev, en*pfd(en))

        mob = np.sum((en*pfd(en))[:,np.newaxis]
            *tau*de, axis=0) / np.sum(fd(en)*de)

        mob = mob *2 /self.effmass

        return mob * self.at2cmvs

    def get_freq(self):
        q = self.qlen / self.bohr
        freq = self.fwq * self.hartree * 1000
        return q, freq
