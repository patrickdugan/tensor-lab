"""Stdlib-only receipt-source regression; no Qiskit or IBM access."""
import unittest
from pathlib import Path
import qpu04_calibration as Q

class HistoricalSources(unittest.TestCase):
    def test_archived_round_and_point_estimates(self):
        a=Q.read_epoch(Q.OLD,True)
        self.assertEqual(len(a["observations"]),16)
        self.assertEqual(set(a["blocks"]),set(Q.BLOCKS))
        self.assertAlmostEqual(a["observations"]["cz.5_6.alpha.5"]["estimate"],.1831136346847817)
        self.assertAlmostEqual(a["observations"]["sx.23"]["estimate"],-.004911995223938367)
        self.assertEqual(a["blocks"]["qc26a"]["counts_sha256"],"ece6e92e7db54b0b21570cc566106b51988b8f1ee02d47fcb275548b0054f5e8")
        self.assertEqual(len({s["path"] for s in a["sources"]}),len(a["sources"]))

    def test_historical_cannot_be_relabelled_fresh(self):
        with self.assertRaisesRegex(ValueError,"ordering seed"):
            Q.read_epoch(Q.OLD,False)

    def test_unarchived_path_refused(self):
        with self.assertRaises(ValueError):
            Q.read_epoch(Path.cwd().parent,False)

if __name__=="__main__":
    unittest.main()
