"""Offline tests of the QC-26B prospective protocol: freeze -> evaluate on SYNTHETIC DRY RUN counts.

No network, no IBM account, no QPU. The test build is a subset of the committed QC-23 nc_eval circuits (QASM and
calibration snapshot of the real Kingston run); counts are fabricated from the engine's own predictions.
Set QC26B_WRITE_DRYRUN_RECEIPT=1 to (re)write receipts/qc26b_prospective_dryrun.json.
Run from this directory:  python -m unittest test_qc26b_prospective -v   (node must be installed)
"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import qc26b_prospective_plan as plan  # noqa: E402
from qc26b_synth_counts import synth_counts  # noqa: E402


def find_node():
    for c in (os.environ.get('QC26B_NODE'), shutil.which('node'), 'C:/Users/patri/AppData/Local/Programs/nodejs/node.exe'):
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    return None


NODE = find_node()
PARAMS = {'sx': {'3': 0.02, '5': 0.04, '23': -0.03},
          'cz': {'16_23': {'delta': 0.05, 'alpha': {'23': 0.02}}, '4_5': {'delta': -0.04}}}


def run(*args):
    return subprocess.run([NODE, *map(str, args)], capture_output=True, text=True, cwd=HERE)


def write_lf(path, text):
    Path(path).write_bytes(text.replace('\r\n', '\n').encode())


def make_subset_build(dst: Path):
    """LF-normalised build dir from the committed nc_eval run: width 8 (all 16 circuits) + width 12 depth<=2 seeds 41,42."""
    meta = json.loads((RES / 'hw_root/compiled/compiled_nc_eval.json').read_text())
    rows = []
    (dst / 'circuits').mkdir(parents=True)
    for r in meta['circuits']:
        if r['n'] == 8 or (r['n'] == 12 and r['depth'] <= 2 and r['seed'] in (41, 42)):
            q = (RES / 'ibm_run_nc/circuits' / (r['id'] + '.qasm')).read_bytes().replace(b'\r\n', b'\n')
            assert hashlib.sha256(q).hexdigest() == r['qasm3_sha256'], r['id']
            (dst / 'circuits' / (r['id'] + '.qasm')).write_bytes(q)
            rows.append(r)
    out = {'set': 'nc_eval_subset', 'backend': 'ibm_kingston', 'mode': 'test-subset', 'shots': 2048,
           'calibration_timestamp': meta['calibration_timestamp'], 'plan_sha256': '0' * 64, 'circuits': rows}
    write_lf(dst / 'compiled.json', json.dumps(out, indent=2, sort_keys=True) + '\n')
    snap = (RES / 'hw_root/compiled/calibration_snapshot_nc_eval.json').read_bytes().replace(b'\r\n', b'\n')
    (dst / 'calibration_snapshot.json').write_bytes(snap)
    return rows


@unittest.skipIf(NODE is None, 'node not available')
class ProspectivePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='qc26b_prosp_'))
        cls.build = cls.tmp / 'build'
        cls.rows = make_subset_build(cls.build)
        cls.params = cls.tmp / 'params.json'
        write_lf(cls.params, json.dumps(PARAMS, indent=1) + '\n')
        cls.frozen_path = cls.tmp / 'frozen.json'
        r = run(HERE / 'qc26b_freeze.mjs', 'freeze', cls.build, cls.params, cls.frozen_path)
        assert r.returncode == 0, r.stderr
        cls.frozen = json.loads(cls.frozen_path.read_text())
        for model in ('C3', 'C0'):
            c = synth_counts(cls.frozen, model, seed=11)
            write_lf(cls.tmp / f'counts_{model}.json', json.dumps(c, sort_keys=True) + '\n')
        cls.reports = {}
        for model in ('C3', 'C0'):
            out = cls.tmp / f'report_{model}.json'
            r = cls.evaluate(cls.frozen_path, cls.build, cls.params, cls.tmp / f'counts_{model}.json', out)
            assert r.returncode == 0, r.stdout + r.stderr
            cls.reports[model] = json.loads(out.read_text())
            cls.reports[model + '_text'] = r.stdout

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def evaluate(frozen, build, params, counts, out):
        return run(HERE / 'qc26b_prospective_eval.mjs', '--frozen', frozen, '--build', build, '--params', params,
                   '--counts', counts, '--out', out)

    def verify(self, frozen, build=None, params=None):
        return run(HERE / 'qc26b_freeze.mjs', 'verify', build or self.build, params or self.params, frozen)

    def rehash(self, doc):
        """Rewrite the top-level sha of a tampered frozen document with the JS canonical hash."""
        body = {k: v for k, v in doc.items() if k != 'sha256'}
        js = ("import {canonical,sha256} from '%s';import fs from 'node:fs';"
              "const b=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));console.log(sha256(canonical(b)));"
              % (HERE / 'qc26b_freeze.mjs').as_uri())
        f = self.tmp / 'body.json'
        f.write_text(json.dumps(body))
        r = subprocess.run([NODE, '--input-type=module', '-e', js, str(f)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return {**body, 'sha256': r.stdout.strip()}

    def test_freeze_contents_and_determinism(self):
        f = self.frozen
        self.assertEqual(len(f['predictions']), len(self.rows))
        self.assertEqual(f['params'], PARAMS)
        self.assertEqual(f['code']['engine_sha256'], hashlib.sha256((HERE / 'qc26b_engine.mjs').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
        for cid, sites in f['predictions'].items():
            info = f['build']['circuits'][cid]
            self.assertEqual(len(sites), info['n'])
            self.assertEqual([s['physical'] for s in sites], info['physical_path'])
            for s in sites:
                for k in ('ideal', 'C0', 'C1', 'C2', 'C3'):
                    self.assertTrue(-1.0001 <= s[k] <= 1.0001)
        again = self.tmp / 'frozen2.json'
        self.assertEqual(run(HERE / 'qc26b_freeze.mjs', 'freeze', self.build, self.params, again).returncode, 0)
        self.assertEqual(json.loads(again.read_text())['sha256'], f['sha256'])
        self.assertEqual(self.verify(self.frozen_path).returncode, 0)
        # C0 ignores parameters; C3 must differ from C2 on circuits that cross the parametrised edges
        gaps = [abs(s['C3'] - s['C2']) for sites in f['predictions'].values() for s in sites]
        self.assertGreater(max(gaps), 1e-3)

    def test_tampered_parameter_file_detected(self):
        bad = copy.deepcopy(PARAMS)
        bad['sx']['23'] += 1e-6
        p = self.tmp / 'params_bad.json'
        write_lf(p, json.dumps(bad) + '\n')
        r = self.verify(self.frozen_path, params=p)
        self.assertEqual(r.returncode, 2)
        self.assertIn('parameter file', r.stdout)
        r = self.evaluate(self.frozen_path, self.build, p, self.tmp / 'counts_C3.json', self.tmp / 'x.json')
        self.assertEqual(r.returncode, 2)
        self.assertIn('REFUSED', r.stderr)

    def test_changed_predictions_detected(self):
        doc = json.loads(self.frozen_path.read_text())
        cid = sorted(doc['predictions'])[0]
        doc['predictions'][cid][0]['C3'] += 1e-6
        p = self.tmp / 'frozen_edit.json'
        p.write_text(json.dumps(doc))
        r = self.verify(p)                                             # contents no longer match the top-level hash
        self.assertEqual(r.returncode, 2)
        self.assertIn('self-hash', r.stdout)
        p.write_text(json.dumps(self.rehash(doc)))                     # forged hash: recomputation catches it
        r = self.verify(p)
        self.assertEqual(r.returncode, 2)
        self.assertIn('predictions recompute', r.stdout)
        r = self.evaluate(p, self.build, self.params, self.tmp / 'counts_C3.json', self.tmp / 'x.json')
        self.assertEqual(r.returncode, 2)
        self.assertIn('REFUSED', r.stderr)
        # a change below the 1e-12 tolerance is accepted
        doc2 = json.loads(self.frozen_path.read_text())
        doc2['predictions'][cid][0]['C3'] += 1e-14
        p.write_text(json.dumps(self.rehash(doc2)))
        self.assertEqual(self.verify(p).returncode, 0)

    def test_changed_engine_hash_or_qasm_detected(self):
        doc = json.loads(self.frozen_path.read_text())
        doc['code']['engine_sha256'] = '0' * 64
        p = self.tmp / 'frozen_engine.json'
        p.write_text(json.dumps(self.rehash(doc)))
        r = self.verify(p)
        self.assertEqual(r.returncode, 2)
        self.assertIn('engine code hash', r.stdout)
        other = self.tmp / 'build_qasm'
        shutil.copytree(self.build, other)
        f = sorted((other / 'circuits').glob('*.qasm'))[0]
        f.write_bytes(f.read_bytes() + b'// tampered\n')
        r = self.verify(self.frozen_path, build=other)
        self.assertEqual(r.returncode, 2)
        self.assertIn('qasm hash', r.stdout)

    def test_crlf_working_copy_still_verifies(self):
        other = self.tmp / 'build_crlf'
        shutil.copytree(self.build, other)
        for f in [*other.glob('circuits/*.qasm'), other / 'compiled.json', other / 'calibration_snapshot.json']:
            f.write_bytes(f.read_bytes().replace(b'\n', b'\r\n'))
        pc = self.tmp / 'params_crlf.json'
        pc.write_bytes(self.params.read_bytes().replace(b'\n', b'\r\n'))
        self.assertEqual(self.verify(self.frozen_path, build=other, params=pc).returncode, 0)

    def test_counts_following_c3_pass(self):
        rep = self.reports['C3']
        self.assertTrue(rep['synthetic'])
        self.assertIn('SYNTHETIC DRY RUN', rep['label'])
        self.assertIn('SYNTHETIC DRY RUN', self.reports['C3_text'])
        c3 = rep['analysis']['candidates']['C3']
        self.assertEqual({k: v['pass'] for k, v in c3['criteria'].items()},
                         {'a_relative_improvement_ge_10pct': True, 'b_paired_ci_lower_positive': True,
                          'c_no_other_site_degradation_gt_0p005': True, 'd_ledger_bound_below_1pct_of_mae': True,
                          'e_no_tuning_freeze_verified': True})
        self.assertEqual(c3['verdict'], 'PASS')
        self.assertGreater(c3['relative_mae_improvement']['pooled_all_sites'], 0.10)
        self.assertGreater(c3['paired_bootstrap_vs_C0']['all_sites']['ci95'][0], 0)
        self.assertEqual(c3['paired_bootstrap_vs_C0']['all_sites']['resamples'], 20000)
        self.assertLess(c3['numerical_vs_physical']['ratio_candidate_min_bound_to_mae'], 0.01)
        sites = {s['physical'] for s in rep['analysis']['sites']}
        self.assertTrue({3, 5, 21, 23} <= sites)           # targets and controls reported per physical site
        self.assertEqual(rep['observables'], sum(r['n'] for r in self.rows))
        self.assertTrue(all('ledger' in r for r in rep['rows']))
        # the report must be deterministic (fixed bootstrap seed)
        out = self.tmp / 'again.json'
        r = self.evaluate(self.frozen_path, self.build, self.params, self.tmp / 'counts_C3.json', out)
        self.assertEqual(json.loads(out.read_text())['analysis'], rep['analysis'])
        if os.environ.get('QC26B_WRITE_DRYRUN_RECEIPT'):
            self.write_receipt()

    def write_receipt(self):
        rep = self.reports['C3']
        keep = {k: v for k, v in rep.items() if k != 'rows'}
        receipt = {'label': 'DRY RUN - SYNTHETIC COUNTS - NOT HARDWARE DATA - pipeline check only',
                   'description': 'freeze -> evaluate on counts fabricated from engine C3 predictions plus Bernoulli shot noise (seed 11), '
                                  'on a subset of the committed QC-23 nc_eval circuits (width 8 all 16, width 12 depth<=2 seeds 41,42) with an '
                                  'ARBITRARY parameter file chosen only to exercise the code. The control run draws counts from C0.',
                   'parameters_used_for_dry_run': PARAMS, 'node': 'v24', 'unittest': 'test_qc26b_prospective.py',
                   'counts_follow_C3': keep,
                   'counts_follow_C0': {k: v for k, v in self.reports['C0'].items() if k not in ('rows',)}}
        path = HERE / 'receipts' / 'qc26b_prospective_dryrun.json'
        path.write_bytes((json.dumps(receipt, indent=1, sort_keys=True) + '\n').encode())

    def test_counts_following_c0_fail_criterion_a(self):
        rep = self.reports['C0']
        c3 = rep['analysis']['candidates']['C3']
        self.assertFalse(c3['criteria']['a_relative_improvement_ge_10pct']['pass'])
        self.assertLess(c3['relative_mae_improvement']['pooled_all_sites'], 0.10)
        self.assertLess(c3['relative_mae_improvement']['targets_5_23'], 0.0)
        self.assertEqual(c3['verdict'], 'FAIL')
        self.assertIn('NEGATIVE RESULT', self.reports['C0_text'])
        self.assertIn('FAIL', self.reports['C0_text'])

    def test_malformed_counts_rejected(self):
        c = json.loads((self.tmp / 'counts_C3.json').read_text())
        cid = sorted(c)[0]
        bad = copy.deepcopy(c)
        bad[cid]['physical_path'] = list(reversed(bad[cid]['physical_path']))
        p = self.tmp / 'counts_bad_path.json'
        p.write_text(json.dumps(bad))
        self.assertNotEqual(self.evaluate(self.frozen_path, self.build, self.params, p, self.tmp / 'x.json').returncode, 0)
        bad = copy.deepcopy(c)
        del bad[cid]
        p.write_text(json.dumps(bad))
        self.assertNotEqual(self.evaluate(self.frozen_path, self.build, self.params, p, self.tmp / 'x.json').returncode, 0)
        bad = copy.deepcopy(c)
        k = next(iter(bad[cid]['counts']))
        bad[cid]['counts'][k] += 1                                       # shot total no longer matches the build
        p.write_text(json.dumps(bad))
        self.assertNotEqual(self.evaluate(self.frozen_path, self.build, self.params, p, self.tmp / 'x.json').returncode, 0)


class PathSearch(unittest.TestCase):
    """Constrained beam search on the committed Kingston calibration snapshot (offline)."""

    @classmethod
    def setUpClass(cls):
        snap = json.loads((RES / 'hw_root/compiled/calibration_snapshot_nc_eval.json').read_text())
        cls.graph = plan.graph_from_snapshot(snap)
        cls.priors, cls.prior_seeds = plan.prior_paths()
        cls.sel = plan.select_paths(cls.graph, priors=cls.priors)

    def test_seeds_are_fresh(self):
        self.assertFalse(set(plan.SEEDS) & self.prior_seeds)
        self.assertEqual(self.prior_seeds, {41, 42, 43, 44})

    def test_constraints_hold(self):
        for (n, seed), rec in self.sel.items():
            path = rec['physical_path']
            self.assertEqual(len(path), n)
            self.assertEqual(len(set(path)), n)
            self.assertNotIn(tuple(path), set(self.priors))
            for a, b in zip(path, path[1:]):
                self.assertIn(b, self.graph[0][a])
            cls = rec['path_class']
            if cls in ('controls', 'pure_control'):
                self.assertFalse({5, 23} & set(path))
            else:
                self.assertTrue({5, 23} <= set(path))
                for q in (5, 23):
                    t = rec['targets'][str(q)]
                    self.assertTrue(t['position_differs_from_all_prior'] or t['neighbours_differ_from_all_prior'])
            if cls == 'targets_end':
                self.assertTrue({3, 21} <= set(path))
                self.assertEqual(path.index(5), 0)
            if cls == 'targets_23_end':
                self.assertIn(path[-1] if path[-1] == 23 else path[0], (23,))
                self.assertNotIn(path.index(5), (0, n - 1))
            if cls == 'controls':
                self.assertIn(21, path)
            if cls == 'pure_control':
                self.assertFalse({3, 4, 5, 6, 16, 21, 22, 23, 24} & set(path))

    def test_all_four_class_types_and_determinism(self):
        classes = {rec['path_class'] for rec in self.sel.values()}
        self.assertEqual(classes, {'targets_end', 'targets_23_end', 'controls', 'pure_control'})
        again = plan.select_paths(self.graph, priors=self.priors)
        self.assertEqual({k: v['physical_path'] for k, v in again.items()}, {k: v['physical_path'] for k, v in self.sel.items()})

    def test_infeasible_constraints_raise(self):
        nbr, edges, readout, oneq = self.graph
        with self.assertRaises(RuntimeError):
            plan.search_line(nbr, edges, readout, oneq, 8, include=[3, 21], exclude=[5, 23])

    def test_submit_cannot_bypass_authorization(self):
        import qc23_access
        ok, why = qc23_access.check_authorization(None, backend='ibm_kingston', sha='x', estimated_seconds=1.0)
        self.assertFalse(ok)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):                         # no build, no frozen file: nothing can run
                plan.submit(Path(d), None, Path(d) / 'f.json', Path(d) / 'p.json')


if __name__ == '__main__':
    unittest.main()
