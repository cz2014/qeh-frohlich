"""
Frohlich electron-phonon coupling calculations for heterostructures.

This module calculates Frohlich coupling potentials that describe electron-phonon
interactions in van der Waals heterostructures. It bridges phonon properties from
building blocks with electron scattering calculations.

Key class:
- BBset: Manages building blocks and computes dynamical matrices and Frohlich potentials
  - get_bb(): Loads phonon data (Born charges, force constants, masses) from .npz files
  - get_dyn_ij(): Computes dynamical matrix elements between layers i and j
  - get_dyn(): Assembles full dynamical matrix for entire heterostructure
  - get_frohlich(): Calculates Frohlich coupling potentials V(q) in meV
    - Modes: 'all' (fully isolated), 'ph' (phonon-normalized), False (coupled)
  - save_froh(): Saves Frohlich data to compressed .npz format

The Frohlich potential V(q,nu) describes how phonon mode nu at wavevector q
scatters electrons, crucial for calculating phonon-limited carrier mobility.
"""

import numpy as np
import ase.units as units
from scipy.interpolate import interp1d

class BBset():
    """ store all the eigenvector and diag of dynamic matrix """

    def __init__(self, path='./'):
        self.bbfiles = {}
        self.path = path

        self.bbdct = {}
        # store data
        # 'qlen': in Bohr^-1 unit, [nq]
        # 'f2q': in Hartree^2 unit, [nq, \sum_k 3N_k]
        # 'froh': in Hartree unit [nq, nmode, nlayer]

    def get_bb(self, layer):
        if layer in self.bbfiles:
            return self.bbfiles[layer]
        
        from . import find_data_file
        mat = layer.split('+')[0]
        dct = np.load(find_data_file(mat + '-phonons.npz'))
        Z_avv = dct['Z_avv']
        C_NN = dct['C_NN']
        m_a = dct['masses']
        cell_cv = dct['cell']

        # Calculate eigenfrequencies
        Minv_NN = np.diag(np.repeat(1 / m_a, 3)**0.5)
        D_NN = np.dot(Minv_NN, np.dot(C_NN, Minv_NN))
        freq2_w, D_xw = np.linalg.eigh(D_NN, UPLO='U')
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        Z_vx = Z_avv.swapaxes(0, 1).reshape((3, -1))
        f2_w = freq2_w * s*s / units.Hartree**2
        m_x = np.repeat(m_a * units._amu / units._me, 3)**0.5

        vol = abs(np.linalg.det(cell_cv)) / units.Bohr**3
        L = np.abs(cell_cv[2, 2] / units.Bohr)
        area = vol / L 

        r"""calculate eigen polarization:
        \sum_{a} q_x \cdot Z^{*}_{a} \cdot \frac{e_a}{\sqrt{M_a}}"""
        pol = np.zeros((len(f2_w),3))
        for i in range(len(f2_w)):
            DM_x = D_xw.T[i] / m_x
            pol[i] = np.dot(Z_vx, DM_x)

        newbb = {}
        newbb['Z_vx'] = Z_vx
        newbb['f2_w'] = f2_w 
        newbb['D_xw'] = D_xw
        newbb['m_x'] = m_x
        newbb['area'] = area
        newbb['pol'] = pol

        self.bbfiles[layer] = newbb
        return newbb

    def get_dyn_ij(self, qlen, Einv_ij, ilayer, jlayer, isdiag=True):
        """qlen in A^{-1}"""

        ibb = self.get_bb(ilayer)
        jbb = self.get_bb(jlayer)
        idim = len(ibb['f2_w'])
        jdim = len(jbb['f2_w'])

        q_abs = qlen * units.Bohr
        dyn_qij = np.zeros((len(q_abs), idim, jdim), dtype=complex)

        if isdiag:
            dyn_qij += np.diagflat(ibb['f2_w'])

        iarea, jarea = ibb['area'], jbb['area']
        ipol, jpol = ibb['pol'], jbb['pol']
        for i in range(idim):
            if i < 3: continue
            for j in range(jdim):
                if j < 3: continue

                dyn_qij[:,i,j] += 1./(iarea*jarea)**0.5 * Einv_ij*q_abs**2 *\
                    ipol[i,0] * jpol[j,0]

                # if i == 5 and j == 5:
                #     pass

        return dyn_qij

    def get_dyn(self, qlen, Einv, layers):
        """ qlen in A^{-1}, Einv[nq,(nw,),nlayer*2, nlayer*2] """

        nq = len(qlen)
        layers = [str(s).replace('+lattpol', '') for s in layers]
        Einv = Einv[:,::2,::2]

        dyn = np.zeros((nq, len(layers), len(layers)), dtype=object)
        for i, ilayer in enumerate(layers):
            for j, jlayer in enumerate(layers):
                isdiag = True if i == j else False
                einv = (Einv[:,i,j] + Einv[:,j,i])/2.0
                dynqij = self.get_dyn_ij(qlen, einv, ilayer, 
                    jlayer, isdiag)
                for iq in range(nq):
                    dyn[iq,i,j] = dynqij[iq]

        dyn = np.array([np.block(dyn[iq,:,:].tolist()) 
            for iq in range(nq)])

        self.dyn = np.real(dyn)
        return np.real(dyn)

    def get_frohlich(self, qlen, Einv, layers, isolated=True, coeff=True):
        """ return potq[nq, mode_tot, nlayer], ignore interlayer coupling if 
        isolated is True"""

        def eigq_sorted(dynq):
            f2q = np.zeros((len(dynq), len(dynq[0])))
            eigq = np.zeros((np.shape(dynq)))
            for i, dyn in enumerate(dynq):
                f2_us, eig_us = np.linalg.eig(dyn)
                f2_us, eig_us = np.real(f2_us), np.real(eig_us)
                f2q[i] = np.sort(f2_us)
                eigq[i] = eig_us.T[np.argsort(f2_us),:]
            return f2q, eigq # f2q in Hartree^2 unit

        def eigq_reordered(dynq):
            f2q = np.zeros((len(dynq), len(dynq[0])))
            eigq = np.zeros((np.shape(dynq)))
            for i, dyn in enumerate(dynq):
                f2_us, eig_us = np.linalg.eig(dyn)
                f2_us, eig_us = np.real(f2_us), np.real(eig_us)
                f2q[i] = np.sort(f2_us)
                eigq[i] = eig_us.T[np.argsort(f2_us),:]
                if i>0:
                    ind = [ np.argmax(
                        [np.abs(np.dot(steig,iqeig)) for iqeig in eigq[i]] ) 
                        for steig in eigq[0] ]
                    f2q[i] = f2q[i][ind]
                    eigq[i] = eigq[i][ind]
            return f2q, eigq # f2q in Hartree^2 unit


        if isolated == "all":
            nq = len(qlen)
            layers = [str(s).replace('+lattpol', '') for s in layers]
            Einv = Einv[:,::2,::2]
            froh = np.zeros((nq, len(layers), len(layers)), dtype=object)
            f2q = []
            for i, ilayer in enumerate(layers):
                dynqii = self.get_dyn_ij(qlen, Einv[:,i,i], ilayer, ilayer)
                f2qm, eigq = eigq_reordered(dynqii)
                f2q.append(f2qm)

                if coeff:
                    eigq = eigq / np.abs(f2qm[:,:,np.newaxis])**0.25 / 2**0.5 

                ibb = self.get_bb(ilayer)
                ipol0 = ibb['pol']
                idim = len(ibb['f2_w'])
                iarea = ibb['area']

                qEinvii = Einv[:,i,i] * qlen * units.Bohr / iarea
                for j, jlayer in enumerate(layers):
                    if i != j: 
                        ifrohm = np.zeros((nq, idim))
                    else:
                        ipolm = np.array([np.dot(eig, ipol0[:,0]) for eig in 
                            eigq])
                        ifrohm = np.multiply(ipolm, 
                            np.repeat(qEinvii[:,np.newaxis], idim, axis=1))
                    for iq in range(nq):
                        froh[iq,i,j] = ifrohm[iq][:,np.newaxis]

            f2q = np.concatenate(f2q, axis=1)
            froh=np.array([np.block(froh[iq,:,:].tolist()) for iq in range(nq)])
        elif isolated == "ph":
            nq = len(qlen)
            layers = [str(s).replace('+lattpol', '') for s in layers]
            Einv = Einv[:,::2,::2]
            froh = np.zeros((nq, len(layers), len(layers)), dtype=object)
            f2q = []
            for i, ilayer in enumerate(layers):
                dynqii = self.get_dyn_ij(qlen, Einv[:,i,i], ilayer, ilayer)
                f2qm, eigq = eigq_reordered(dynqii)
                f2q.append(f2qm)

                if coeff:
                    eigq = eigq / np.abs(f2qm[:,:,np.newaxis])**0.25 / 2**0.5 

                ibb = self.get_bb(ilayer)
                ipol0 = ibb['pol']
                idim = len(ibb['f2_w'])
                iarea = ibb['area']

                for j, jlayer in enumerate(layers):
                    jarea = self.get_bb(jlayer)['area']
                    qEinvij = Einv[:,i,j] * qlen * units.Bohr / (
                        jarea*iarea)**0.5 # ph norm!
                    ipolm = np.array([np.dot(eig, ipol0[:,0]) for eig in eigq])
                    ifrohm = np.multiply(ipolm, 
                            np.repeat(qEinvij[:,np.newaxis], idim, axis=1))
                    for iq in range(nq):
                        froh[iq,i,j] = ifrohm[iq][:,np.newaxis]

            f2q = np.concatenate(f2q, axis=1)
            froh=np.array([np.block(froh[iq,:,:].tolist()) for iq in range(nq)])
        else:
            nq = len(qlen)
            q_abs = qlen * units.Bohr
            dynq = self.get_dyn(qlen, Einv, layers)
            layers = [str(s).replace('+lattpol', '') for s in layers]
            Einv = Einv[:,::2,::2]

            f2q, eigq = eigq_reordered(dynq)

            if coeff:
                eigq = eigq / np.abs(f2q[:,:,np.newaxis])**0.25 / 2**0.5 

            froh = []
            areas = np.array([self.get_bb(l)['area'] for l in layers])
            for i, ilayer in enumerate(layers):
                ibb = self.get_bb(ilayer)
                ipol0 = ibb['pol'][:,0]
                idim = len(ibb['f2_w'])
                iarea = ibb['area']

                qEinvi = np.multiply(Einv[:,i,:], np.repeat(q_abs[:,np.newaxis],
                    len(layers), axis=1))
                qEinvi = np.repeat(qEinvi[:,np.newaxis,:], idim, axis=1) 
                qEinvi = qEinvi / (
                    areas[np.newaxis, np.newaxis, :]*iarea)**0.5 # ph norm!
                ifroh = ipol0[np.newaxis,:,np.newaxis]*qEinvi
                froh.append(np.copy(ifroh))
            froh = np.concatenate(froh, axis=1)
            froh = np.array([np.dot(eigq[iq],froh[iq]) for iq in range(nq)])
            # eigq[nq,nmode,nmode], froh[nq,nmode,nlayer]

        self.bbdct['qlen'] = qlen * units.Bohr
        self.bbdct['f2q'] = f2q
        self.bbdct['froh'] = froh
        if hasattr(self, 'dyn') and self.dyn is not None:
            self.bbdct['dyn'] = self.dyn

        return np.real(froh)*units.Hartree*1000 # in meV unit

    def save_froh(self, filename):
        assert bool(self.bbdct) is True
        np.savez_compressed(filename, **self.bbdct)

