"""Offline ideal-circuit tests. No account, service, or hardware calls."""
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qc26a_compile import load_plan,logical_circuit,assert_counts,estimate

class QC26A(unittest.TestCase):
    def test_all_logical_circuits(self):
        plan=load_plan()
        for row in plan['rows']:
            with self.subTest(id=row['id']):
                qc=logical_circuit(row);assert_counts(row,qc)
                n=len(row['path']);u=QuantumCircuit(n)
                for inst in qc.data:
                    if inst.operation.name not in ('barrier','measure'):
                        u.append(inst.operation,[qc.find_bit(q).index for q in inst.qubits])
                probs=Statevector.from_instruction(u).probabilities()
                idx=np.arange(len(probs))
                z=[np.sum(probs*(1-2*((idx>>q)&1))) for q in range(n)]
                f=row['family']
                want=([1-2*row['state']]*n if f=='spam' else
                      [float(row['axis']=='Z')]*n if f=='sx' else
                      [1-2*row['state'],float(row['axis']=='X')])
                np.testing.assert_allclose(z,want,atol=1e-10,rtol=0)
    def test_budget(self):
        plan=load_plan()
        rows=[{**r,'duration_s':10e-6} for r in plan['rows']]
        self.assertAlmostEqual(estimate(plan,rows),16.38688,places=7)
        self.assertLess(estimate(plan,rows),20)
        self.assertGreater(estimate(plan,[{**r,'duration_s':1e-3} for r in plan['rows']]),20)

if __name__=='__main__':unittest.main()
