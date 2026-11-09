"""QC-13 exact-dyadic and symbolic validation (no unverified global claims)."""
from __future__ import annotations
import unittest
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np
from channel_mpo import I2,X,Y,Z,kraus,gate_layers
from qc12_heisenberg import operator_cone_expectation
from qc13_dyadic import (residual_certificate,svd_stored_certificate,
                         kraus_completeness_certificate,mpo_zero_readout_certificate)
from qc13_rational import exact_circuit_expectation
from qc13_clifford import (HH,CNOT,CZ,SWAP,S_I,TABLES,GATES,
                           clifford_layers,exact_clifford_expectation)
from qc13_audit import audited_observable
from qc11_streaming_cone import streaming_cone_density


class ExactDyadicTests(unittest.TestCase):
    def test_exact_squared_complex_residual(self):
        m=np.array([[1+2j,.25-1j],[-.125j,0]],complex)
        l=np.array([[.5-1j],[1+0j]],complex)
        r=np.array([[2+.25j,-.5j]],complex)
        res=residual_certificate(m,l,r)
        total=F(0)
        for i in range(2):
            for j in range(2):
                z=m[i,j]; a=l[i,0];b=r[0,j]
                re=F(z.real)-(F(a.real)*F(b.real)-F(a.imag)*F(b.imag))
                im=F(z.imag)-(F(a.real)*F(b.imag)+F(a.imag)*F(b.real))
                total+=re*re+im*im
        self.assertEqual(res.squared_exact(),total)
        self.assertGreaterEqual(F(res.upper)**2,total)

    def test_exact_stored_rounding_and_nonfinite(self):
        rng=np.random.default_rng(19)
        m=rng.normal(size=(24,18))+1j*rng.normal(size=(24,18))
        u,s,vh=np.linalg.svd(m,full_matrices=False)
        for rank in (2,8,18):
            cert=svd_stored_certificate(m,u,s,vh,rank)
            # This NumPy norm is an empirical cross-check only; the cert is
            # proved by the integer algorithm independently of this assertion.
            nominal=np.linalg.norm(m-u[:,:rank]@(s[:rank,None]*vh[:rank]))
            self.assertLess(abs(cert.upper-nominal),1e-11)
            self.assertGreaterEqual(F(cert.upper)**2,cert.squared_exact())
        with self.assertRaises(ValueError):residual_certificate(m,np.full((24,1),np.nan),np.ones((1,18)))
        with self.assertRaises(ValueError):residual_certificate(np.eye(3),np.ones((2,2)),np.eye(2))

    def test_subnormal_and_cancellation(self):
        tiny=np.nextafter(0.,1.)
        m=np.array([[1.,tiny],[tiny,-1.]],complex)
        l=np.array([[1.],[tiny]],complex)
        r=np.array([[1.,0.]],complex)
        cert=residual_certificate(m,l,r)
        self.assertGreater(cert.upper,1.)
        self.assertGreaterEqual(F(cert.upper)**2,cert.squared_exact())

    def test_readout_exact_negative_and_imaginary(self):
        a=np.zeros((1,2,2,1),complex);a[0,0,0,0]=-.125+.25j
        b=np.zeros((1,2,2,1),complex);b[0,0,0,0]=.3-.4j
        c=mpo_zero_readout_certificate([a,b])
        exact=(-.125+.25j)*(.3-.4j)
        self.assertLessEqual(c.real_lo,exact.real+1e-16)
        self.assertGreaterEqual(c.real_hi,exact.real-1e-16)
        self.assertLessEqual(c.imaginary_lo,exact.imag+1e-16)
        self.assertGreaterEqual(c.imaginary_hi,exact.imag-1e-16)
        self.assertGreaterEqual(c.maximum_real_error(exact.real),0)
        with self.assertRaises(ValueError):mpo_zero_readout_certificate([])

    def test_exact_completeness_defect(self):
        self.assertEqual(kraus_completeness_certificate((I2,)),0.)
        for name in ('depolarizing','dephasing','amplitude_damping'):
            r=kraus_completeness_certificate(kraus(name,.2))
            self.assertGreaterEqual(r,0.)
            self.assertLess(r,1e-14)
        with self.assertRaises(ValueError):kraus_completeness_certificate((np.eye(3),))


class OracleTests(unittest.TestCase):
    def test_full_exact_float_circuit_oracle(self):
        for name in ('depolarizing','dephasing','amplitude_damping'):
            ops=kraus(name,.5)
            layers=gate_layers(3,3,3)
            exact=exact_circuit_expectation(3,layers,ops,(0,),'Z')
            numeric=operator_cone_expectation(3,layers,ops,(0,),'Z',bond_cap=16)
            self.assertLess(exact.absolute_difference(numeric.expectation),2e-12)
            self.assertLess(abs(float(exact.trace_real_error)),1e-12)
            self.assertLessEqual(exact.expectation_real_lo,exact.expectation_real_hi)

    def test_all_clifford_maps_exact_in_rationals(self):
        self.assertEqual(set(TABLES),set(GATES))
        for name,table in TABLES.items():
            self.assertEqual(len(table),16)
            self.assertEqual(table[('I','I')],(('I','I'),1))
            self.assertEqual(len({value[0] for value in table.values()}),16)
        self.assertTrue(np.array_equal(HH.conj().T@HH,np.eye(4)))

    def test_clifford_exact_vs_dense_all_channels(self):
        n=8;layers=clifford_layers(n,6,5)
        gates=[[(i,u) for i,u,_ in layer] for layer in layers]
        for name,p in (('depolarizing',F(1,5)),('dephasing',F(1,2)),
                       ('amplitude_damping',F(1,2))):
            ex=exact_clifford_expectation(n,layers,name,p,0)
            rho=streaming_cone_density(n,gates,kraus(name,float(p)),(0,),max_sites=8).reduced_density
            val=float(np.trace(rho@Z).real)
            self.assertLess(ex.absolute_difference(val),1e-11)

    def test_deep_48_qubit_exact_channel_oracle(self):
        n=48;layers=clifford_layers(n,16,5)
        gates=[[(i,u) for i,u,_ in layer] for layer in layers]
        for name in ('dephasing','depolarizing','amplitude_damping'):
            p=F(1,2)
            ex=exact_clifford_expectation(n,layers,name,p,22,max_terms=1000)
            approx=operator_cone_expectation(n,gates,kraus(name,float(p)),(22,),'Z',bond_cap=16)
            self.assertLess(ex.absolute_difference(approx.expectation),approx.error_bound+1e-10)
            self.assertLess(ex.absolute_difference(approx.expectation),1e-9)
        with self.assertRaises(ValueError):exact_clifford_expectation(n,layers,'unknown',F(1,2),22)

    def test_invalid_oracle_inputs_fail_closed(self):
        with self.assertRaises(ValueError):exact_circuit_expectation(4,[],kraus('dephasing',.2),(0,),'Z')
        with self.assertRaises(ValueError):exact_circuit_expectation(2,[],kraus('dephasing',.2),(0,),'W')
        with self.assertRaises(ValueError):clifford_layers(1,4,0)
        with self.assertRaises(ValueError):exact_clifford_expectation(2,[],'dephasing',F(1,2),2)
        # A hard cap prevents uncontrolled expansion in amplitude damping.
        layers=clifford_layers(48,16,5)
        with self.assertRaises(MemoryError):
            exact_clifford_expectation(48,layers,'amplitude_damping',F(1,2),22,max_terms=1)


class AuditedMPOTests(unittest.TestCase):
    def test_audit_matches_numerical_ledger(self):
        layers=gate_layers(8,6,3);ops=kraus('amplitude_damping',.5)
        baseline=operator_cone_expectation(8,layers,ops,(0,),'Z',bond_cap=8)
        audited=audited_observable(8,layers,ops,(0,),'Z',bond_cap=8)
        self.assertLess(abs(audited.expectation-baseline.expectation),1e-12)
        self.assertLess(abs(audited.local_residual_upper-baseline.error_bound),1e-10)
        self.assertLess(audited.arithmetic_readout_gap(),1e-12)
        self.assertEqual(audited.local_certified_count,len(baseline.local_residuals))
        self.assertGreaterEqual(F(audited.local_residual_upper),audited.local_fraction_exact_sum)
        self.assertLess(audited.gate_coeff_defect,1e-12)
        self.assertLess(audited.kraus_completeness_defect,1e-12)

    def test_audit_keeps_dense_oracle_out(self):
        layers=gate_layers(24,8,5);ops=kraus('dephasing',.5)
        with patch('qc11_streaming_cone.streaming_cone_density',side_effect=RuntimeError('oracle')):
            a=audited_observable(24,layers,ops,(12,),'Z',bond_cap=8)
        self.assertEqual(a.cone_width,16)
        self.assertLess(a.arithmetic_readout_gap(),1e-12)


class AcceptanceTests(unittest.TestCase):
    def test_exact_clifford_selector_weak_amplitude_damping(self):
        from qc13_acceptance import threshold_clifford
        layers=clifford_layers(48,16,3)
        r=threshold_clifford(48,layers,'amplitude_damping',F(1,5),22,
                             caps=(1,2,4,8),max_error=.05)
        self.assertTrue(r.accepted)
        self.assertEqual(r.chosen_cap,2)
        self.assertLess(r.error_upper,.05)
        with self.assertRaises(ValueError):
            threshold_clifford(48,layers,'amplitude_damping',F(1,5),22,caps=(8,2))

    def test_universal_physical_range_is_independent_of_truncation(self):
        from qc13_acceptance import physical_range_bound,physical_pauli_range
        self.assertEqual(physical_pauli_range('amplitude_damping',F(1,2),'Z'),(F(0),F(1)))
        self.assertEqual(physical_pauli_range('depolarizing',F(1,5),'Z'),(-F(4,5),F(4,5)))
        layers=gate_layers(8,6,3)
        a=audited_observable(8,layers,kraus('amplitude_damping',.2),(0,),'Z',bond_cap=2)
        upper=physical_range_bound('amplitude_damping',F(1,5),'Z',a.readout)
        oracle=exact_circuit_expectation(3,gate_layers(3,3,3),kraus('amplitude_damping',.2),(0,),'Z')
        self.assertTrue(upper>0)
        self.assertLess(upper,3)
        self.assertLessEqual(oracle.expectation_real_lo,1+1e-12)
        with self.assertRaises(ValueError):physical_pauli_range('foo',F(1,2),'Z')


if __name__=='__main__':unittest.main()
