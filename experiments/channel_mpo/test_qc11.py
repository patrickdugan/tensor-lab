"""QC-11 regression suite for streamed exact local-cone evolution."""
import unittest
from unittest.mock import patch
import numpy as np
import qc11_streaming_cone as sc
from qc11_streaming_cone import streaming_cone_density,streaming_region_bundle
from qc11_focus import target_schedule,cone_weights,ObservablePeakChain
from qc10_cone import exact_cone_density,trace_distance,reduced_density
from channel_mpo import gate_layers,kraus,apply_operator,apply_channel
from qc09_certified import CertifiedPeakChain


class StreamingConeTests(unittest.TestCase):
    def test_three_channels_two_regions_vs_qc10(self):
        n=8
        for channel in ('depolarizing','dephasing','amplitude_damping'):
            layers=gate_layers(n,6,seed=5)
            for p in (.2,.5):
                ops=kraus(channel,p)
                for target in ((0,),(0,1),(6,7)):
                    a=streaming_cone_density(n,layers,ops,target)
                    b=exact_cone_density(n,layers,ops,target,max_sites=8)
                    self.assertLess(trace_distance(a.reduced_density,b.reduced_density),2e-12)
                    self.assertLess(abs(np.trace(a.reduced_density)-1),2e-12)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(a.reduced_density).min()),-2e-12)

    def test_large_chain_dephasing_failure_recovered_exactly(self):
        n=48
        for seed,target in ((3,(n-2,n-1)),(7,(0,1))):
            layers=gate_layers(n,6,seed)
            ops=kraus('dephasing',.5)
            a=streaming_cone_density(n,layers,ops,target)
            b=exact_cone_density(n,layers,ops,target)
            self.assertLess(trace_distance(a.reduced_density,b.reduced_density),2e-12)
            self.assertEqual(a.per_layer_widths,(8,7,6,5,4,3,2))

    def test_small_n_against_full_dense(self):
        n=5
        for channel in ('depolarizing','amplitude_damping'):
            layers=gate_layers(n,3,seed=2)
            ops=kraus(channel,.25)
            rho=np.zeros((2**n,2**n),complex);rho[0,0]=1
            for layer in layers:
                for i,u in layer:rho=apply_operator(rho,u,(i,i+1),n)
                for i in range(n):rho=apply_channel(rho,ops,i,n)
            for target in ((0,),(0,1),(3,4)):
                expected=reduced_density(rho,n,target)
                actual=streaming_cone_density(n,layers,ops,target).reduced_density
                self.assertLess(trace_distance(expected,actual),2e-12)

    def test_bundle_reuses_nested_regions(self):
        n=48;layers=gate_layers(n,6,seed=3);ops=kraus('dephasing',.5)
        regions={'left_1':(0,), 'left_2':(0,1),'right_1':(n-1,),'right_2':(n-2,n-1)}
        results,solves=streaming_region_bundle(n,layers,ops,regions)
        self.assertEqual(len(solves),2)
        self.assertEqual(set(results),set(regions))
        self.assertLess(trace_distance(reduced_density(results['left_2'],2,(0,)),results['left_1']),1e-12)
        self.assertLess(trace_distance(reduced_density(results['right_2'],2,(1,)),results['right_1']),1e-12)
        self.assertLess(trace_distance(exact_cone_density(n,layers,ops,regions['right_2']).reduced_density,
                                       results['right_2']),2e-12)

    def test_only_small_matrices_ever_evolved(self):
        n=48;layers=gate_layers(n,6,seed=3);ops=kraus('dephasing',.5)
        widths=[]
        orig_op=sc.apply_operator;orig_ch=sc.apply_channel
        def op(rho,gate,target,m):
            widths.append(m);self.assertLessEqual(m,8)
            return orig_op(rho,gate,target,m)
        def channel(rho,k,target,m):
            widths.append(m);self.assertLessEqual(m,8)
            return orig_ch(rho,k,target,m)
        with patch.object(sc,'apply_operator',side_effect=op),patch.object(sc,'apply_channel',side_effect=channel):
            result=streaming_cone_density(n,layers,ops,(46,47))
        self.assertEqual(result.max_density_entries,4**8)
        self.assertEqual(result.peak_density_entries,4**8)
        self.assertTrue(widths)
        self.assertLess(result.kraus_site_updates,6*8)

    def test_fail_early_on_dense_memory_cap(self):
        n=48;layers=gate_layers(n,6,3);ops=kraus('depolarizing',.2)
        with self.assertRaises(MemoryError):
            streaming_cone_density(n,layers,ops,(46,47),max_density_entries=4**8-1)
        result=streaming_cone_density(n,layers,ops,(46,47),max_density_entries=4**8)
        self.assertEqual(result.max_density_entries,4**8)
        with self.assertRaises(ValueError):
            streaming_cone_density(n,layers,ops,(46,47),max_sites=7)

    def test_validation_rejects_nonphysical_inputs(self):
        layers=gate_layers(8,6,seed=3)
        with self.assertRaises(ValueError):
            streaming_cone_density(8,layers,(np.diag([1,.8]),),(0,1))
        with self.assertRaises(ValueError):
            streaming_cone_density(8,layers,kraus('dephasing',.2),(0,1,1))
        bad=[list(layer) for layer in layers]
        i,u=bad[0][0];bad[0][0]=(i,2*u)
        with self.assertRaises(ValueError):
            streaming_cone_density(8,bad,kraus('dephasing',.2),(0,1))
        with self.assertRaises(ValueError):
            streaming_region_bundle(8,layers,kraus('dephasing',.2),{'invalid':(1,0)})

    def test_causality_in_noiseless_empty_gates(self):
        n=8;ops=kraus('dephasing',.5);layers=[[],[],[]]
        for target in ((0,),(3,4)):
            res=streaming_cone_density(n,layers,ops,target)
            expected=np.zeros_like(res.reduced_density);expected[0,0]=1
            self.assertLess(trace_distance(res.reduced_density,expected),1e-12)
            self.assertEqual(len(res.cone.sites),len(target))

    def test_topology_scores_depend_only_on_geometry(self):
        n=48;layers=gate_layers(n,6,3)
        a=target_schedule(n,layers,((0,1),(46,47)))
        self.assertEqual(len(a.active_after),6)
        self.assertEqual(len(a.active_after[-1]),4)
        bonds,env=cone_weights(n,a.active_after[-1],off_cone=.02)
        self.assertEqual(env[0],1.)
        self.assertEqual(env[24],.02)
        self.assertEqual(bonds[1],1.)
        self.assertEqual(bonds[20],.02)
        self.assertEqual(bonds[45],1.)
        with self.assertRaises(ValueError):
            cone_weights(n,(0,1),off_cone=0)

    def test_unweighted_allocator_matches_qc09_baseline(self):
        n=6;layers=gate_layers(n,2,seed=3);ops=kraus('dephasing',.5)
        schedule=target_schedule(n,layers,((0,1),))
        baseline=CertifiedPeakChain(n,max_bond=12,max_env=12,work_cap_entries=750000,
                      gate_method='sketch',oversample=4,power_iters=2,
                      peak_gate_cap_entries=260000)
        focus=ObservablePeakChain(n,max_bond=12,max_env=12,work_cap_entries=750000,
                      gate_method='sketch',oversample=4,power_iters=2,
                      peak_gate_cap_entries=260000)
        for t,layer in enumerate(layers):
            for x in (baseline,focus):
                x.gate_layer(layer)
                for site in range(n):x.channel(site,ops)
            a=baseline.compress_layer_peak((96,416)[t],ops)
            b=focus.compress_focused((96,416)[t],ops,schedule,t,off_cone=1.0)
            self.assertEqual(a.plan.joint.bond_ranks,b.plan.joint.bond_ranks)
            self.assertEqual(a.plan.joint.environment_ranks,b.plan.joint.environment_ranks)
        from qc10_cone import tensor_marginal
        self.assertLess(trace_distance(tensor_marginal(baseline,(0,1)),
                                       tensor_marginal(focus,(0,1))),1e-12)

    def test_empty_and_duplicate_bundle_rejected(self):
        n=8;layers=gate_layers(n,1,3);ops=kraus('dephasing',.2)
        with self.assertRaises(ValueError):
            streaming_region_bundle(n,layers,ops,{})
        with self.assertRaises(ValueError):
            streaming_region_bundle(n,layers,ops,{'a':(-1,)})


if __name__=='__main__':unittest.main()
