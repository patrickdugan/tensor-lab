"""QC-12 numerical and analytical-ledger regression tests."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np
from channel_mpo import X,Y,Z,I2,kraus,gate_layers
from qc10_cone import trace_distance,reduced_density
from qc11_streaming_cone import streaming_cone_density
from qc12_heisenberg import (_independent_residual,OperatorMPO,
                                operator_cone_expectation,pauli_density,threshold_observable,threshold_density)


class BackwardMPOTests(unittest.TestCase):
    def test_exact_single_pauli_across_channels(self):
        for channel in ('depolarizing','dephasing','amplitude_damping'):
            ops=kraus(channel,.25)
            for target,label in (((0,),'Z'),((4,),'Y'),((3,),'X')):
                n=9;layers=gate_layers(n,6,2)
                obs=operator_cone_expectation(n,layers,ops,target,label,bond_cap=128)
                rho=streaming_cone_density(n,layers,ops,target,max_sites=9).reduced_density
                exact=float(np.trace(rho@{'X':X,'Y':Y,'Z':Z}[label]).real)
                self.assertLess(abs(exact-obs.expectation),3e-12)
                self.assertLess(obs.error_bound,2e-10)
                self.assertEqual(obs.cone_width,obs.per_layer_widths[-1])

    def test_truncated_expectations_respect_recorded_ideal_bounds(self):
        for seed in (0,2,5):
            layers=gate_layers(8,6,seed)
            for channel in ('depolarizing','dephasing','amplitude_damping'):
                ops=kraus(channel,.3)
                for cap in (1,2,4):
                    result=operator_cone_expectation(8,layers,ops,(0,1),'XZ',bond_cap=cap)
                    reference=streaming_cone_density(8,layers,ops,(0,1)).reduced_density
                    expected=float(np.trace(reference@np.kron(X,Z)).real)
                    self.assertLessEqual(abs(expected-result.expectation),result.error_bound+1e-10)
                    self.assertTrue(all(r.frobenius_residual>=0 for r in result.local_residuals))

    def test_small_density_tomography_bound(self):
        n=8;layers=gate_layers(n,6,3)
        for channel in ('dephasing','amplitude_damping'):
            ops=kraus(channel,.5)
            expected=streaming_cone_density(n,layers,ops,(0,1)).reduced_density
            for cap in (2,8):
                approximate,bound,parts=pauli_density(n,layers,ops,(0,1),bond_cap=cap)
                self.assertEqual(len(parts),16)
                self.assertLessEqual(trace_distance(approximate,expected),bound+1e-10)
                self.assertLess(abs(np.trace(approximate)-1),1e-9)
                self.assertTrue(np.allclose(approximate,approximate.conj().T))

    def test_deep_causal_cone_avoids_dense_oracle(self):
        n=48;layers=gate_layers(n,16,3)
        with patch('qc11_streaming_cone.streaming_cone_density',side_effect=RuntimeError('dense oracle')):
            result=operator_cone_expectation(n,layers,kraus('dephasing',.5),(22,),'Z',bond_cap=8)
        self.assertEqual(result.cone_width,32)
        self.assertEqual(result.per_layer_widths[0],1)
        self.assertEqual(result.per_layer_widths[-1],32)
        self.assertLess(result.max_mpo_entries,10000)
        self.assertLess(result.error_bound,.05)

    def test_deeper_boundary_circuit_and_noiseless_analytic(self):
        n=48;depth=24;layers=gate_layers(n,depth,3)
        result=operator_cone_expectation(n,layers,kraus('dephasing',.5),(0,),'Z',bond_cap=8)
        self.assertEqual(result.cone_width,24)
        self.assertLess(result.error_bound,.05)
        empty=[[] for _ in range(30)]
        for channel in ('dephasing','amplitude_damping','depolarizing'):
            val=operator_cone_expectation(n,empty,kraus(channel,.2),(17,),'Z',bond_cap=1)
            expected=1. if channel!='depolarizing' else .8**len(empty)
            self.assertAlmostEqual(val.expectation,expected,places=10)
            self.assertLess(val.error_bound,1e-12)

    def test_svd_independent_residual(self):
        rng=np.random.default_rng(14)
        m=rng.standard_normal((19,13))+1j*rng.standard_normal((19,13))
        u,s,v=np.linalg.svd(m,full_matrices=False)
        for r in (1,5,13):
            exact=np.linalg.norm(m-(u[:,:r]*s[:r])@v[:r])
            self.assertAlmostEqual(_independent_residual(m,u,s,v,r,block=3),exact,places=12)

    def test_validation_and_entry_limits(self):
        layers=gate_layers(8,5,0);ops=kraus('dephasing',.2)
        fn=lambda **kwargs:operator_cone_expectation(8,layers,ops,(0,),'Z',**kwargs)
        with self.assertRaises(ValueError):fn(bond_cap=0)
        with self.assertRaises(MemoryError):fn(max_cone=2)
        with self.assertRaises(MemoryError):fn(max_entries=3)
        with self.assertRaises(ValueError):operator_cone_expectation(8,layers,ops,(0,2),'ZZ')
        with self.assertRaises(ValueError):operator_cone_expectation(8,layers,ops,(0,),'W')
        with self.assertRaises(ValueError):operator_cone_expectation(8,layers,(np.eye(2)*.5,),(0,),'Z')
        with self.assertRaises(ValueError):pauli_density(8,layers,ops,(0,1,2))
        with self.assertRaises(ValueError):fn(local_tol=-1)
        bad=[[tuple(g) for g in layer] for layer in layers]
        i,u=bad[0][0];bad[0][0]=(i,u*2)
        with self.assertRaises(ValueError):operator_cone_expectation(8,bad,ops,(0,),'Z')

    def test_predeclared_threshold_selector_does_not_use_oracle(self):
        n=48;layers=gate_layers(n,8,5);ops=kraus('dephasing',.5)
        with patch('qc11_streaming_cone.streaming_cone_density',side_effect=RuntimeError('oracle called')):
            d=threshold_density(n,layers,ops,(0,1),caps=(4,8,16),max_error=.05)
        self.assertTrue(d.accepted)
        self.assertEqual(d.chosen_cap,8)
        self.assertLess(d.bound,.05)
        oracle=streaming_cone_density(n,layers,ops,(0,1),max_sites=10)
        self.assertLessEqual(trace_distance(d.estimate,oracle.reduced_density),d.bound+1e-10)
        with patch('qc11_streaming_cone.streaming_cone_density',side_effect=RuntimeError('oracle called')):
            z=threshold_observable(n,layers,ops,(0,),'Z',caps=(2,4,8),max_error=.001)
        if z.accepted:
            true=streaming_cone_density(n,layers,ops,(0,),max_sites=10).reduced_density
            self.assertLessEqual(abs(z.estimate-float(np.trace(true@Z).real)),z.bound+1e-10)

    def test_threshold_rejection_and_invalid_caps(self):
        n=8;layers=gate_layers(n,5,3);ops=kraus('dephasing',.5)
        decision=threshold_observable(n,layers,ops,(0,),'Z',caps=(1,),max_error=0.)
        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.estimate)
        for caps in ((8,4),(4,4),()):
            with self.assertRaises(ValueError):
                threshold_density(n,layers,ops,(0,),caps=caps)
        with self.assertRaises(ValueError):
            threshold_observable(n,layers,ops,(0,),'Z',max_error=-.1)

    def test_mpo_nontrivial_identity_padding_and_decomposition(self):
        mpo=OperatorMPO((3,4),'XZ',10)
        mpo.grow(1,6)
        self.assertEqual((mpo.first,mpo.last,mpo.width),(1,6,6))
        # An identity gate still permits QR/SVD normalization and no truncation.
        loss=mpo.gate_adjoint(3,np.eye(4),layer=1)
        self.assertLess(loss,1e-12)
        self.assertEqual(mpo.width,6)
        self.assertAlmostEqual(mpo.input_zero_expectation(),0.)


if __name__=='__main__':unittest.main()
