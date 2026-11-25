import unittest
from qc23_ibm import echo_schedule, inverse_gate, plan, selected_bits, z_estimators

class QC23Test(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(echo_schedule(8,4,3), echo_schedule(8,4,3))
        self.assertNotEqual(echo_schedule(8,4,3), echo_schedule(8,4,5))
    def test_op_inversion(self):
        self.assertEqual(inverse_gate('sx'),'sxdg')
        self.assertEqual(inverse_gate('cx'),'cx')
    def test_bounds(self):
        for n in (2,8,16,48):
            self.assertEqual(len(set(selected_bits(n))),len(selected_bits(n)))
            self.assertTrue(all(0<=v<n for v in selected_bits(n)))
    def test_counts(self):
        z=z_estimators({'000':9,'001':1},3,(0,1,2))
        self.assertAlmostEqual(z['0']['expectation_z'],.8)
        self.assertAlmostEqual(z['1']['expectation_z'],1.)
    def test_run_manifest(self):
        self.assertEqual(len(plan((8,16),(2,4,8),(3,5))),12)
        self.assertEqual(plan()[0]['shots'],2048)
    def test_bad_input(self):
        with self.assertRaises(ValueError):echo_schedule(1,2,3)
        with self.assertRaises(ValueError):z_estimators({},3,(1,))

if __name__=='__main__':unittest.main()
