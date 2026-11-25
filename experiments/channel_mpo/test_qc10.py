"""QC-10 independent exact light-cone and operational-error regression tests."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np

from channel_mpo import gate_layers, kraus, apply_operator, apply_channel, X, Z
from qc10_cone import (backward_cone, exact_cone_density, reduced_density,
                       tensor_marginal, trace_distance, local_comparison)
from qc09_certified import CertifiedPeakChain
from benchmark_qc10 import run_case


def dense_oracle(n, layers, operators):
    rho=np.zeros((2**n,2**n),complex)
    rho[0,0]=1
    for layer in layers:
        for i,u in layer:rho=apply_operator(rho,u,(i,i+1),n)
        for i in range(n):rho=apply_channel(rho,operators,i,n)
    return rho


class TestQC10(unittest.TestCase):
    def test_backward_cone_boundary_widths(self):
        n=48;layers=gate_layers(n,6,3)
        for target,width in [((0,),6),((0,1),8),((47,),6),((46,47),8)]:
            self.assertEqual(len(backward_cone(n,layers,target).sites),width)
        self.assertEqual(len(backward_cone(n,layers,(24,)).sites),12)

    def test_depth_zero_product_state(self):
        for targets in [(0,), (6,), (0,7)]:
            result=exact_cone_density(48,[],kraus('depolarizing',.5),targets)
            self.assertEqual(len(result.cone.sites),len(targets))
            self.assertAlmostEqual(result.reduced_density[0,0].real,1.)

    def test_strict_inputs_and_cone_limit(self):
        n=48;layers=gate_layers(n,6,3)
        for targets in [(0,0),(),(48,),(-1,),(1,0)]:
            with self.assertRaises(ValueError):backward_cone(n,layers,targets)
        with self.assertRaises(ValueError):exact_cone_density(n,layers,kraus('dephasing',.2),(24,),max_sites=8)
        with self.assertRaises(ValueError):exact_cone_density(n,layers,(np.eye(2)*.5,), (0,))
        with self.assertRaises(ValueError):backward_cone(4,[[(0,np.eye(4)),(1,np.eye(4))]],(0,))

    def test_exact_lightcone_equal_full_dense_all_channels(self):
        n=6
        for channel in ('depolarizing','dephasing','amplitude_damping'):
            for seed in (0,3):
                gates=gate_layers(n,6,seed);ops=kraus(channel,.3)
                complete=dense_oracle(n,gates,ops)
                for support in ((0,),(1,),(0,1),(4,5),(0,5)):
                    oracle=exact_cone_density(n,gates,ops,support,max_sites=6)
                    expected=reduced_density(complete,n,support)
                    self.assertLess(trace_distance(oracle.reduced_density,expected),2e-12)

    def test_nontrivial_crop_eight_qubit(self):
        n=8; gates=gate_layers(n,6,4);ks=kraus('amplitude_damping',.3)
        full=dense_oracle(n,gates,ks)
        a=exact_cone_density(n,gates,ks,(0,))
        self.assertEqual(len(a.cone.sites),6)
        self.assertLess(trace_distance(a.reduced_density,reduced_density(full,n,(0,))),1e-12)

    def test_irrelevant_gate_outside_cone(self):
        n=8;layers=gate_layers(n,6,2);ops=kraus('depolarizing',.2)
        a=exact_cone_density(n,layers,ops,(0,))
        changed=[list(x) for x in layers]
        changed[0]=[(i,(np.kron(X,Z) if i==6 else u)) for i,u in changed[0]]
        b=exact_cone_density(n,changed,ops,(0,))
        self.assertLess(trace_distance(a.reduced_density,b.reduced_density),1e-14)

    def test_tensor_marginal_matches_untruncated_dense(self):
        n=4;ops=kraus('dephasing',.3);layers=gate_layers(n,3,3)
        chain=CertifiedPeakChain(n,max_bond=64,max_env=64,work_cap_entries=500_000,gate_method='full')
        rho=np.zeros((2**n,2**n),complex);rho[0,0]=1.
        for layer in layers:
            chain.gate_layer(layer)
            for i,u in layer:rho=apply_operator(rho,u,(i,i+1),n)
            for i in range(n):
                chain.channel(i,ops);rho=apply_channel(rho,ops,i,n)
            chain.compress_layer_peak(50_000,ops)
            chain.normalize()
        for support in [(0,),(3,),(0,1),(0,3)]:
            self.assertLess(trace_distance(tensor_marginal(chain,support),
                           reduced_density(rho,n,support)),2e-10)

    def test_tensor_marginal_validation(self):
        chain=CertifiedPeakChain(6)
        for targets in [(),(1,1),(2,1),(-1,),(7,),tuple(range(6))]:
            with self.assertRaises(ValueError):tensor_marginal(chain,targets)

    def test_operational_trace_distance_bound(self):
        ref=np.diag([1.,0.]).astype(complex)
        pert=np.diag([.9,.1]).astype(complex)
        class Fake:pass
        o=Fake();o.reduced_density=ref
        result=local_comparison(o,pert)
        self.assertAlmostEqual(result['trace_distance'],.1)
        self.assertAlmostEqual(result['worst_povm_event_error'],.1)
        self.assertAlmostEqual(result['worst_observable_abs_error_bound'],.2)
        self.assertAlmostEqual(result['z_abs_error'],.2)

    def test_small_cone_no_large_density_allocation(self):
        # An 8-site oracle is constructed for a 48-site problem.
        n=48;layers=gate_layers(n,6,3);ops=kraus('depolarizing',.2)
        import qc10_cone
        real=qc10_cone.apply_operator
        observed=[]
        def guard(rho,op,targets,n_sites):
            observed.append(n_sites)
            self.assertLessEqual(n_sites,8)
            return real(rho,op,targets,n_sites)
        with patch.object(qc10_cone,'apply_operator',side_effect=guard):
            result=exact_cone_density(n,layers,ops,(0,1))
        self.assertEqual(result.reduced_density.shape,(4,4))
        self.assertTrue(observed)

    def test_benchmark_accepts_real_48_qubit_results(self):
        out=run_case(48,.2,3,modes=('peak',))
        self.assertEqual(len(out),4)
        self.assertTrue(all(0<=row['local_trace_distance']<=1 for row in out))
        self.assertTrue(all(row['observable_abs_error']<=row['worst_norm1_observable_error_bound']+1e-9 for row in out))
        self.assertTrue(all(row['cone_width']<=8 for row in out))
        self.assertTrue(all(row['retained_entries']>0 for row in out))

    def test_bad_grid_rejected(self):
        with self.assertRaises(ValueError):run_case(48,.2,3,modes=('wrong',))
        with self.assertRaises(ValueError):run_case(48,.2,3,depth=5)


if __name__=='__main__':unittest.main()

class TestQC10Acceptance(unittest.TestCase):
    @staticmethod
    def artificial(peak_d,sketch_d):
        base={'n':48,'p':.2,'seed':3,'depth':6,'channel':'depolarizing',
              'modeled_live_peak_entries':100,'retained_entries':80,'sim_seconds':1.}
        return [{**base,'mode':mode,'region':region,'local_trace_distance':distance}
                for mode,distance in (('peak',peak_d),('sketch',sketch_d))
                for region in ('left_1','left_2','right_1','right_2')]

    def test_peak_preferred_within_tolerance(self):
        from qc10_acceptance import select_acceptable
        case=select_acceptable(self.artificial(.045,.02),.05)
        self.assertEqual(case[0]['decision'],'peak')

    def test_sketch_fallback_if_peak_fails(self):
        from qc10_acceptance import select_acceptable
        case=select_acceptable(self.artificial(.055,.04),.05)
        self.assertEqual(case[0]['decision'],'sketch')
        self.assertLessEqual(case[0]['max_d_selected'],.05)

    def test_reject_if_no_mode_meets_bound(self):
        from qc10_acceptance import select_acceptable
        case=select_acceptable(self.artificial(.07,.06),.05)
        self.assertEqual(case[0]['decision'],'reject')

    def test_missing_or_duplicate_region_rejected(self):
        from qc10_acceptance import select_acceptable
        rows=self.artificial(.03,.02)
        with self.assertRaises(ValueError):select_acceptable(rows[:-1],.05)
        with self.assertRaises(ValueError):select_acceptable(rows+[rows[0]],.05)

    def test_bad_tolerance_rejected(self):
        from qc10_acceptance import select_acceptable
        with self.assertRaises(ValueError):select_acceptable(self.artificial(.03,.02),float('nan'))
        with self.assertRaises(ValueError):select_acceptable(self.artificial(.03,.02),-1.)

class TestQC10DisjointCones(unittest.TestCase):
    def test_output_factorization_against_small_full_density(self):
        from qc10_cone import independent_cones_density
        n=8;layers=gate_layers(n,2,5);ops=kraus('dephasing',.25)
        full=dense_oracle(n,layers,ops)
        expected=reduced_density(full,n,(0,7))
        factor=independent_cones_density(n,layers,ops,(0,),(7,))
        self.assertLess(trace_distance(expected,factor),1e-12)

    def test_overlapping_cones_raise(self):
        from qc10_cone import independent_cones_density
        layers=gate_layers(8,6,3)
        with self.assertRaises(ValueError):
            independent_cones_density(8,layers,kraus('depolarizing',.2),(0,),(7,))

    def test_48_qubit_ends_have_disjoint_cones(self):
        from qc10_cone import independent_cones_density
        layers=gate_layers(48,6,3)
        joint=independent_cones_density(48,layers,kraus('depolarizing',.2),(0,),(47,))
        self.assertEqual(joint.shape,(4,4))
        self.assertAlmostEqual(np.trace(joint).real,1.,places=12)
