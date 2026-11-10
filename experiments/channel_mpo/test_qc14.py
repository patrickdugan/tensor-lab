"""QC-14 end-to-end binary64 auditor tests against independent references."""
from __future__ import annotations
import unittest
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np

from channel_mpo import I2,Z,kraus,gate_layers
from qc10_cone import exact_cone_density
from qc13_rational import exact_circuit_expectation,qmatrix,dot,dagger,kron
from qc13_clifford import CZ,clifford_layers,exact_clifford_expectation
from qc14_certified import (two_site_residual,channel_local_residual,
    certify_observable,_polar_dual_defect,_tensor_norm_bound)


class DyadicOperationChecks(unittest.TestCase):
    def test_unchanged_pair_and_changed_site(self):
        a=np.arange(16).reshape(2,2,2,2).astype(complex)/8
        b=np.arange(16).reshape(2,2,2,2).astype(complex)/16
        self.assertEqual(two_site_residual(a,b,a.copy(),b.copy()),0.)
        c=a.copy();c[0,0,0,0]+=.125
        self.assertGreater(two_site_residual(a,b,c,b),0.)
        self.assertGreater(_tensor_norm_bound(a),0.)

    def test_exact_pauli_gate_pair(self):
        a=Z.reshape(1,2,2,1);b=I2.reshape(1,2,2,1)
        # CZ commutes with Z⊗I, and CZ is an exact-dyadic Clifford.
        self.assertEqual(two_site_residual(a,b,a,b,unitary=CZ),0.)
        x=np.array([[0,1],[1,0]],complex).reshape(1,2,2,1)
        self.assertGreater(two_site_residual(x,b,x,b,unitary=CZ),0.)

    def test_rounding_and_channel_dyadic_reference(self):
        old=np.zeros((1,2,2,1),complex);old[0,:,:,0]=Z
        operators=kraus('dephasing',.2)
        exact_like=np.einsum('ab,lbcq,cd->ladq',operators[0].conj().T,old,operators[0])
        for k in operators[1:]:
            exact_like+=np.einsum('ab,lbcq,cd->ladq',k.conj().T,old,k)
        bound=channel_local_residual(old,exact_like,operators)
        self.assertLess(bound,1e-14)
        damaged=exact_like.copy();damaged[0,0,0,0]+=.1
        self.assertGreater(channel_local_residual(old,damaged,operators),.099)
        with self.assertRaises(ValueError):
            channel_local_residual(old,np.ones((1,3,3,1)),operators)

    def test_polar_correction_defect(self):
        self.assertEqual(_polar_dual_defect(0.),0.)
        self.assertGreaterEqual(_polar_dual_defect(.01),.02)
        with self.assertRaises(ValueError):_polar_dual_defect(.8)


class EndToEndNumericalAudit(unittest.TestCase):
    def test_exact_literal_dyadic_oracle_all_noise_channels(self):
        for ch in ('depolarizing','dephasing','amplitude_damping'):
            with self.subTest(channel=ch):
                layers=gate_layers(3,3,3);ks=kraus(ch,.5)
                oracle=exact_circuit_expectation(3,layers,ks,(0,),'Z')
                for cap in (1,2,4):
                    r=certify_observable(3,layers,ks,(0,),'Z',bond_cap=cap)
                    # The exact oracle uses literal coefficients; the audited
                    # ideal target polar-normalizes them. Their discrepancy is
                    # several orders smaller than the reported enclosure here.
                    self.assertLess(oracle.absolute_difference(r.reported_expectation),
                                    r.reported_error_upper + 1e-13)
                    self.assertLess(r.readout_rounding_upper,2e-14)
                    self.assertGreaterEqual(r.reported_error_upper,r.physical_error_upper)
                    self.assertEqual(sum(r.event_counts.values()),len(r.events))
                self.assertLess(certify_observable(3,layers,ks,(0,),'Z',bond_cap=4)
                                .reported_error_upper,1e-11)

    def test_exact_dense_cone_nontrivial_short_certificate(self):
        layers=gate_layers(10,6,3);ks=kraus('dephasing',.5)
        r=certify_observable(10,layers,ks,(0,),'Z',bond_cap=4)
        oracle=exact_cone_density(10,layers,ks,(0,),max_sites=8)
        correct=float(np.trace(oracle.reduced_density@Z).real)
        self.assertLess(abs(correct-r.reported_expectation),r.reported_error_upper)
        self.assertLess(r.reported_error_upper,.05)
        self.assertGreater(r.event_counts['left_qr'],0)
        self.assertGreater(r.event_counts['right_qr'],0)

    def test_qr_corruption_widens_bound(self):
        layers=gate_layers(6,3,1);ops=kraus('dephasing',.5)
        baseline=certify_observable(6,layers,ops,(0,),'Z',bond_cap=8)
        real_qr=np.linalg.qr
        def corrupt(*args,**kw):
            q,r=real_qr(*args,**kw)
            return q * 1.001,r
        with patch('qc14_certified.np.linalg.qr',side_effect=corrupt):
            perturbed=certify_observable(6,layers,ops,(0,),'Z',bond_cap=8)
        self.assertLess(baseline.reported_error_upper,1e-10)
        self.assertGreater(perturbed.reported_error_upper,.001)

    def test_nonunitary_stored_gate_polar_target_is_bounded(self):
        # Deliberately perturb a stored gate. Its polar completion is the
        # identity, so the ideal Z_0 expectation on |00> is exactly 1.
        faulty=np.diag([1.01,1.,1.,1.]).astype(complex)
        r=certify_observable(2,[[(0,faulty)]],(I2,),(0,),'Z',bond_cap=4)
        actual_error=abs(r.reported_expectation-1.)
        self.assertGreater(actual_error,.02)
        self.assertGreaterEqual(r.reported_error_upper,actual_error)
        self.assertGreater(r.polar_correction_sum_upper,.04)

    def test_fail_closed(self):
        l=gate_layers(6,3,3);ks=kraus('dephasing',.5)
        with self.assertRaises(MemoryError):
            certify_observable(6,l,ks,(0,),'Z',bond_cap=4,max_events=1)
        with self.assertRaises(MemoryError):
            certify_observable(6,l,ks,(0,),'Z',bond_cap=4,max_cone=1)
        with self.assertRaises(ValueError):
            certify_observable(6,l,(np.zeros((2,2),complex),),(0,),'Z')
        with self.assertRaises(ValueError):
            certify_observable(6,l,ks,(0,),'W')

if __name__=='__main__':unittest.main()
