"""QC-10 independent causal-cone and local operational-error tests."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np
from channel_mpo import gate_layers,kraus,apply_operator,apply_channel,X,Z
from qc10_cone import (backward_cone,exact_cone_density,reduced_density,
                       tensor_marginal,trace_distance,local_comparison,
                       independent_cones_density)
from qc09_certified import CertifiedPeakChain
from benchmark_qc10 import run_case
from qc10_acceptance import select_acceptable

def dense_oracle(n,layers,operators):
    rho=np.zeros((2**n,2**n),complex)
    rho[0,0]=1.
    for layer in layers:
        for i,u in layer:rho=apply_operator(rho,u,(i,i+1),n)
        for i in range(n):rho=apply_channel(rho,operators,i,n)
    return rho

class TestQC10(unittest.TestCase):
    def test_backward_cone_boundary_widths(self):
        n=48;layers=gate_layers(n,6,3)
        for targets,width in (((0,),6),((0,1),8),((47,),6),((46,47),8)):
            self.assertEqual(len(backward_cone(n,layers,targets).sites),width)
        self.assertEqual(len(backward_cone(n,layers,(24,)).sites),12)

    def test_depth_zero_product_state(self):
        for targets in ((0,),(6,),(0,7)):
            r=exact_cone_density(48,[],kraus("depolarizing",.5),targets)
            self.assertEqual(len(r.cone.sites),len(targets))
            self.assertAlmostEqual(r.reduced_density[0,0].real,1.)

    def test_strict_inputs_and_cone_limit(self):
        n=48;layers=gate_layers(n,6,3)
        for targets in ((0,0),(),(48,),(-1,),(1,0)):
            with self.assertRaises(ValueError):backward_cone(n,layers,targets)
        with self.assertRaises(ValueError):
            exact_cone_density(n,layers,kraus("dephasing",.2),(24,),max_sites=8)
        with self.assertRaises(ValueError):
            exact_cone_density(n,layers,(np.eye(2)*.5,),(0,))
        with self.assertRaises(ValueError):
            backward_cone(4,[[(0,np.eye(4)),(1,np.eye(4))]],(0,))

    def test_all_channels_against_dense(self):
        n=6
        for channel in ("depolarizing","dephasing","amplitude_damping"):
            for seed in (0,3):
                layers=gate_layers(n,6,seed);ops=kraus(channel,.3)
                dense=dense_oracle(n,layers,ops)
                for target in ((0,),(1,),(0,1),(4,5),(0,5)):
                    cone=exact_cone_density(n,layers,ops,target,max_sites=6)
                    self.assertLess(trace_distance(cone.reduced_density,
                                    reduced_density(dense,n,target)),2e-12)

    def test_nontrivial_eight_qubit_crop(self):
        n=8;layers=gate_layers(n,6,4);ops=kraus("amplitude_damping",.3)
        full=dense_oracle(n,layers,ops)
        r=exact_cone_density(n,layers,ops,(0,))
        self.assertEqual(len(r.cone.sites),6)
        self.assertLess(trace_distance(r.reduced_density,
                        reduced_density(full,n,(0,))),1e-12)

    def test_irrelevant_gate_does_not_affect_cone(self):
        n=8;layers=gate_layers(n,6,2);ops=kraus("depolarizing",.2)
        a=exact_cone_density(n,layers,ops,(0,))
        changed=[list(layer) for layer in layers]
        changed[0]=[(i,(np.kron(X,Z) if i==6 else u)) for i,u in changed[0]]
        b=exact_cone_density(n,changed,ops,(0,))
        self.assertLess(trace_distance(a.reduced_density,b.reduced_density),1e-14)

    def test_tensor_marginal_matches_untruncated_dense(self):
        n=4;ops=kraus("dephasing",.3);layers=gate_layers(n,3,3)
        chain=CertifiedPeakChain(n,max_bond=64,max_env=64,
                work_cap_entries=500000,gate_method="full")
        rho=np.zeros((2**n,2**n),complex);rho[0,0]=1
        for layer in layers:
            chain.gate_layer(layer)
            for i,u in layer:rho=apply_operator(rho,u,(i,i+1),n)
            for i in range(n):
                chain.channel(i,ops)
                rho=apply_channel(rho,ops,i,n)
            chain.compress_layer_peak(50000,ops)
            chain.normalize()
        for targets in ((0,),(3,),(0,1),(0,3)):
            self.assertLess(trace_distance(tensor_marginal(chain,targets),
                            reduced_density(rho,n,targets)),2e-10)

    def test_tensor_marginal_validation(self):
        chain=CertifiedPeakChain(6)
        for targets in ((),(1,1),(2,1),(-1,),(7,),tuple(range(6))):
            with self.assertRaises(ValueError):tensor_marginal(chain,targets)

    def test_operational_trace_distance_bound(self):
        rho=np.diag([1.,0.]).astype(complex)
        approx=np.diag([.9,.1]).astype(complex)
        class Ref: pass
        x=Ref();x.reduced_density=rho
        outcome=local_comparison(x,approx)
        self.assertAlmostEqual(outcome["trace_distance"],.1)
        self.assertAlmostEqual(outcome["worst_povm_event_error"],.1)
        self.assertAlmostEqual(outcome["worst_observable_abs_error_bound"],.2)
        self.assertAlmostEqual(outcome["z_abs_error"],.2)

    def test_no_full_48_qubit_dense_allocation(self):
        import qc10_cone
        layers=gate_layers(48,6,3);ops=kraus("depolarizing",.2)
        real=qc10_cone.apply_operator
        dimensions=[]
        def guard(rho,op,targets,n):
            dimensions.append(n)
            self.assertLessEqual(n,8)
            return real(rho,op,targets,n)
        with patch.object(qc10_cone,"apply_operator",side_effect=guard):
            result=exact_cone_density(48,layers,ops,(0,1))
        self.assertEqual(result.reduced_density.shape,(4,4))
        self.assertTrue(dimensions)

    def test_benchmark_48_qubit_smoke(self):
        records=run_case(48,.2,3,modes=("peak",))
        self.assertEqual(len(records),4)
        for r in records:
            self.assertTrue(0<=r["local_trace_distance"]<=1)
            self.assertLessEqual(r["observable_abs_error"],
                                 r["worst_norm1_observable_error_bound"]+1e-9)
            self.assertLessEqual(r["cone_width"],8)

    def test_bad_grid_rejected(self):
        with self.assertRaises(ValueError):run_case(48,.2,3,modes=("wrong",))
        with self.assertRaises(ValueError):run_case(48,.2,3,depth=5)

class TestQC10Acceptance(unittest.TestCase):
    @staticmethod
    def artificial(peak_d,sketch_d):
        base={"n":48,"p":.2,"seed":3,"depth":6,"channel":"depolarizing",
              "modeled_live_peak_entries":100,"retained_entries":80,"sim_seconds":1.}
        return [{**base,"mode":mode,"region":region,"local_trace_distance":d}
                for mode,d in (("peak",peak_d),("sketch",sketch_d))
                for region in ("left_1","left_2","right_1","right_2")]

    def test_peak_preferred_within_tolerance(self):
        self.assertEqual(select_acceptable(self.artificial(.045,.02),.05)[0]["decision"],"peak")

    def test_sketch_fallback(self):
        row=select_acceptable(self.artificial(.055,.04),.05)[0]
        self.assertEqual(row["decision"],"sketch")
        self.assertLessEqual(row["max_d_selected"],.05)

    def test_reject_both_modes(self):
        self.assertEqual(select_acceptable(self.artificial(.07,.06),.05)[0]["decision"],"reject")

    def test_missing_and_duplicate_region(self):
        cases=self.artificial(.03,.02)
        with self.assertRaises(ValueError):select_acceptable(cases[:-1],.05)
        with self.assertRaises(ValueError):select_acceptable(cases+[cases[0]],.05)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            select_acceptable(self.artificial(.03,.02),float("nan"))
        with self.assertRaises(ValueError):
            select_acceptable(self.artificial(.03,.02),-1.)

class TestQC10DisjointCones(unittest.TestCase):
    def test_exact_boundary_factorization(self):
        n=8;layers=gate_layers(n,2,5);ops=kraus("dephasing",.25)
        full=dense_oracle(n,layers,ops)
        reference=reduced_density(full,n,(0,7))
        factor=independent_cones_density(n,layers,ops,(0,),(7,))
        self.assertLess(trace_distance(reference,factor),1e-12)

    def test_overlapping_cones_raise(self):
        with self.assertRaises(ValueError):
            independent_cones_density(8,gate_layers(8,6,3),
                                      kraus("depolarizing",.2),(0,),(7,))

    def test_48_qubit_boundaries_independent(self):
        joint=independent_cones_density(48,gate_layers(48,6,3),
                                        kraus("depolarizing",.2),(0,),(47,))
        self.assertEqual(joint.shape,(4,4))
        self.assertAlmostEqual(np.trace(joint).real,1.,places=12)

if __name__=="__main__":unittest.main()
