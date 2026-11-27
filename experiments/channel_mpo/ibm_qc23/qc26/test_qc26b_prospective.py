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
# Larger endpoint terms so that C3Z is measurably different from C3 (used for the C3Z tests only).
PARAMS_Z = {'sx': PARAMS['sx'],
            'cz': {'16_23': {'delta': 0.05, 'alpha': {'16': 0.12, '23': 0.15}}, '4_5': {'delta': -0.04, 'alpha': {'4': -0.10, '5': 0.15}}}}
PARAMS_Z0 = {'sx': PARAMS['sx'], 'cz': {k: {'delta': v['delta']} for k, v in PARAMS_Z['cz'].items()}}   # same delta, no alpha
PARAMS_Z_ALPHA0 = {'sx': PARAMS['sx'], 'cz': {k: {'delta': v['delta'], 'alpha': {q: 0.0 for q in v['alpha']}} for k, v in PARAMS_Z['cz'].items()}}


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
        cls.z = {}
        for name, prm in (('Z', PARAMS_Z), ('Z0', PARAMS_Z0), ('Zalpha0', PARAMS_Z_ALPHA0)):
            pf, ff = cls.tmp / f'params_{name}.json', cls.tmp / f'frozen_{name}.json'
            write_lf(pf, json.dumps(prm, indent=1) + '\n')
            r = run(HERE / 'qc26b_freeze.mjs', 'freeze', cls.build, pf, ff)
            assert r.returncode == 0, r.stderr
            cls.z[name] = {'params': pf, 'frozen_path': ff, 'frozen': json.loads(ff.read_text())}
        for model in ('C3Z', 'C3'):
            c = synth_counts(cls.z['Z']['frozen'], model, seed=11)
            write_lf(cls.tmp / f'counts_Z_{model}.json', json.dumps(c, sort_keys=True) + '\n')
            out = cls.tmp / f'report_Z_{model}.json'
            r = cls.evaluate(cls.z['Z']['frozen_path'], cls.build, cls.z['Z']['params'], cls.tmp / f'counts_Z_{model}.json', out)
            assert r.returncode == 0, r.stdout + r.stderr
            cls.reports['Z_' + model] = json.loads(out.read_text())
            cls.reports['Z_' + model + '_text'] = r.stdout
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


    def test_v3_rehashed_metadata_ledger_and_scoring_code_tampering(self):
        cid = sorted(self.frozen['predictions'])[0]
        def changed_ledger(doc):
            doc['predictions'][cid][0]['ledger']['C3']['min_bound'] += 0.01
        def changed_site(doc):
            doc['predictions'][cid][0]['physical'] += 1
        def changed_depth(doc):
            doc['build']['circuits'][cid]['depth'] += 1
        def changed_score_hash(doc):
            doc['code']['manifest']['qc26b_prospective_eval.mjs'] = '0' * 64
        for change in (changed_ledger, changed_site, changed_depth, changed_score_hash):
            doc = copy.deepcopy(self.frozen)
            change(doc)
            p = self.tmp / ('forged_' + change.__name__ + '.json')
            p.write_text(json.dumps(self.rehash(doc)))
            self.assertEqual(self.verify(p).returncode, 2, change.__name__)

    def test_path_clustered_estimand_matches_pooled_mae(self):
        a = self.reports['C3']['analysis']
        pc = a['candidates']['C3']['path_clustered_bootstrap_vs_C0']['all_sites']
        self.assertAlmostEqual(pc['mean_improvement'], a['groups']['all']['C0'] - a['groups']['all']['C3'])
        self.assertEqual(sum(r['n_obs'] for r in pc['per_path']), a['groups']['all']['n'])
        self.assertEqual(pc['clusters'], len({(r['n'], r['seed']) for r in self.rows}))

    def test_legacy_v2_replay_and_live_rejection(self):
        doc = copy.deepcopy(self.frozen)
        doc['format'] = 'qc26b-frozen-v2'
        doc['code'].pop('manifest')
        p = self.tmp / 'legacy_v2.json'
        p.write_text(json.dumps(self.rehash(doc)))
        self.assertEqual(self.verify(p).returncode, 0)
        with self.assertRaisesRegex(ValueError, 'requires v3'):
            plan.verify_published_freeze({}, doc)

    def test_usage_gate_fail_closed(self):
        class Service:
            def __init__(self, usage): self.value = usage
            def usage(self): return self.value
        with self.assertRaises(ValueError):
            plan.checked_usage(Service({}), 15)
        with self.assertRaises(ValueError):
            plan.checked_usage(Service({'usage_remaining_seconds': 14}), 15)
        self.assertEqual(plan.checked_usage(Service({'usage_remaining_seconds': 16}), 15)['usage_remaining_seconds'], 16)

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
                                  'ARBITRARY parameter file chosen only to exercise the code. The control run draws counts from C0. '
                                  'Frozen format qc26b-frozen-v2: C3 uses delta only (alpha ignored); C3Z is the declared extension (C3 plus alpha) and is '
                                  'reported beside C3, never as the primary comparator. counts_follow_C3Z uses a second arbitrary parameter file with '
                                  'larger alpha so that C3Z differs visibly from C3. Regenerated for the C3Z schema: the previous receipt (format v1) '
                                  'had C3 including the alpha term of the parameter file.',
                   'parameters_used_for_dry_run': PARAMS, 'parameters_used_for_C3Z_dry_run': PARAMS_Z,
                   'node': 'v24', 'unittest': 'test_qc26b_prospective.py',
                   'counts_follow_C3': keep,
                   'counts_follow_C0': {k: v for k, v in self.reports['C0'].items() if k not in ('rows',)},
                   'declared_extension_C3Z': {
                       'counts_follow_C3Z': {k: v for k, v in self.reports['Z_C3Z'].items() if k != 'rows'},
                       'counts_follow_C3_under_Z_params': {k: v for k, v in self.reports['Z_C3'].items() if k != 'rows'}}}
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


    # ---- declared extension C3Z -------------------------------------------------------------------------------------
    def sites(self, frozen):
        return [(cid, s) for cid, ss in sorted(frozen['predictions'].items()) for s in ss]

    def test_c3z_equals_c3_when_alpha_zero_or_absent(self):
        for name in ('Z0', 'Zalpha0'):                                   # alpha absent / alpha explicitly 0
            for cid, s in self.sites(self.z[name]['frozen']):
                self.assertEqual(s['C3Z'], s['C3'], (name, cid, s['bit']))
        # PARAMS_Z and its alpha-free copy differ only in C3Z: ideal, C0..C3 are bit-identical
        a, b = self.z['Z']['frozen'], self.z['Z0']['frozen']
        for (cid, sa), (_, sb) in zip(self.sites(a), self.sites(b)):
            for k in ('ideal', 'C0', 'C1', 'C2', 'C3'):
                self.assertEqual(sa[k], sb[k], (cid, k))
            self.assertEqual(sb['C3Z'], sb['C3'])
            self.assertEqual(sa['ledger']['C3'], sb['ledger']['C3'])

    def test_c3_ignores_alpha(self):
        # C3 of the original PARAMS (contains alpha 0.02) equals C3 with alpha removed
        p0 = {'sx': PARAMS['sx'], 'cz': {k: {'delta': v['delta']} for k, v in PARAMS['cz'].items()}}
        pf, ff = self.tmp / 'params_p0.json', self.tmp / 'frozen_p0.json'
        write_lf(pf, json.dumps(p0, indent=1) + '\n')
        self.assertEqual(run(HERE / 'qc26b_freeze.mjs', 'freeze', self.build, pf, ff).returncode, 0)
        other = json.loads(ff.read_text())
        for (cid, sa), (_, sb) in zip(self.sites(self.frozen), self.sites(other)):
            self.assertEqual(sa['C3'], sb['C3'])
            self.assertEqual(sb['C3Z'], sb['C3'])
        self.assertTrue(any(abs(s['C3Z'] - s['C3']) > 1e-4 for _, s in self.sites(self.frozen)))

    def test_c3z_differs_from_c3_when_alpha_nonzero(self):
        gaps = [abs(s['C3Z'] - s['C3']) for _, s in self.sites(self.z['Z']['frozen'])]
        self.assertGreater(max(gaps), 1e-2)
        f = self.z['Z']['frozen']
        self.assertIn('C3Z', f['models'])
        self.assertIn('DECLARED EXTENSION', f['models']['C3Z'])
        self.assertEqual(f['format'], 'qc26b-frozen-v3')
        for cid, ss in f['predictions'].items():
            for s in ss:
                self.assertIn('C3Z', s['ledger'])
                self.assertTrue(-1.0001 <= s['C3Z'] <= 1.0001)
        self.assertEqual(self.verify(self.z['Z']['frozen_path'], params=self.z['Z']['params']).returncode, 0)

    def test_counts_from_c3z_pass_c3z_criteria(self):
        rep = self.reports['Z_C3Z']
        cands = rep['analysis']['candidates']
        z, c3 = cands['C3Z'], cands['C3']
        self.assertEqual(z['verdict'], 'PASS')
        self.assertTrue(all(v['pass'] for v in z['criteria'].values()))
        self.assertIn('DECLARED EXTENSION', z['status'])
        self.assertEqual(set(z['criteria']), set(c3['criteria']))       # same criteria (a)-(e)
        v = z['vs_C3']['paired_bootstrap_MAE_C3_minus_MAE_C3Z']
        self.assertEqual(v['all_sites']['base'], 'C3')
        self.assertEqual(v['all_sites']['resamples'], 20000)
        self.assertGreater(v['all_sites']['ci95'][0], 0)
        self.assertGreater(v['targets_5_23']['ci95'][0], 0)
        self.assertGreater(v['all_sites']['mean_improvement'], 0)
        for s in ('site_5', 'site_23'):
            self.assertLess(z['target_site_mae'][s]['C3Z'], z['target_site_mae'][s]['C3'])
        self.assertLess(z['numerical_vs_physical']['ratio_candidate_min_bound_to_mae'], 0.01)
        self.assertIn('flagged_sites', z['other_site_degradation'])
        a = rep['analysis']['groups']['all']
        self.assertLess(a['C3Z'], a['C3'])
        txt = self.reports['Z_C3Z_text']
        self.assertIn('DECLARED EXTENSION', txt)
        self.assertIn('paired bootstrap MAE(C3)-MAE(C3Z)', txt)
        self.assertIn('target-site MAE', txt)
        self.assertNotIn('(PRIMARY)', txt.split('=== C3Z')[1])
        self.assertIn('=== C3 (PRIMARY)', txt)

    def test_counts_from_c3_are_not_scored_better_by_c3z(self):
        rep = self.reports['Z_C3']
        cands = rep['analysis']['candidates']
        a = rep['analysis']['groups']['all']
        self.assertLess(a['C3'], a['C3Z'])                               # counts drawn from C3: the extension does worse
        v = cands['C3Z']['vs_C3']['paired_bootstrap_MAE_C3_minus_MAE_C3Z']
        self.assertLess(v['all_sites']['mean_improvement'], 0)
        self.assertLess(v['all_sites']['ci95'][1], 0)
        self.assertEqual(cands['C3']['verdict'], 'PASS')                 # C3 is judged on its own, the extension never changes it

    def make_v1(self, v2):
        """Reduce a v2 frozen document to a legacy v1 one: no C3Z; C3 as originally defined (alpha as given in params) = C3Z."""
        d = copy.deepcopy(v2)
        d['format'] = 'qc26b-frozen-v1'
        del d['models']['C3Z']
        d['models']['C3'] = 'C2 + coherent CZ phases (params.cz)'
        for ss in d['predictions'].values():
            for s in ss:
                s['C3'] = s.pop('C3Z')
                s['ledger']['C3'] = s['ledger'].pop('C3Z')
        return self.rehash(d)

    def test_primary_c3_analysis_identical_for_v1_and_v2(self):
        # alpha-free parameters: the v1 reduction and the v2 file give the same C3 and C2 analysis on the same counts
        z0 = self.z['Z0']['frozen']
        p = self.tmp / 'frozen_v1_z0.json'
        p.write_text(json.dumps(self.make_v1(z0)))
        counts = self.tmp / 'counts_Z0.json'
        write_lf(counts, json.dumps(synth_counts(z0, 'C3', seed=5), sort_keys=True) + '\n')
        o1, o2 = self.tmp / 'r_v1.json', self.tmp / 'r_v2.json'
        self.assertEqual(self.evaluate(p, self.build, self.z['Z0']['params'], counts, o1).returncode, 0)
        self.assertEqual(self.evaluate(self.z['Z0']['frozen_path'], self.build, self.z['Z0']['params'], counts, o2).returncode, 0)
        a, b = json.loads(o1.read_text())['analysis'], json.loads(o2.read_text())['analysis']
        for k in ('C3', 'C2'):
            self.assertEqual(a['candidates'][k], b['candidates'][k])
        self.assertNotIn('C3Z', a['candidates'])
        self.assertIn('C3Z', b['candidates'])
        self.assertEqual(b['candidates']['C3Z']['relative_mae_improvement'], b['candidates']['C3']['relative_mae_improvement'])

    def test_legacy_v1_frozen_files_still_verify(self):
        for name in ('Z', 'Z0'):                                         # with alpha (legacy C3 used it) and without
            v1 = self.make_v1(self.z[name]['frozen'])
            p = self.tmp / f'frozen_v1_{name}.json'
            p.write_text(json.dumps(v1))
            r = self.verify(p, params=self.z[name]['params'])
            self.assertEqual(r.returncode, 0, r.stdout)
            counts = self.tmp / f'counts_v1_{name}.json'
            write_lf(counts, json.dumps(synth_counts(v1, 'C3', seed=3), sort_keys=True) + '\n')
            out = self.tmp / f'report_v1_{name}.json'
            r = self.evaluate(p, self.build, self.z[name]['params'], counts, out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rep = json.loads(out.read_text())
            self.assertEqual(set(rep['analysis']['candidates']), {'C3', 'C2'})
            self.assertEqual(rep['analysis']['candidates']['C3']['verdict'], 'PASS')
            self.assertNotIn('C3Z', r.stdout)
        # a tampered legacy file is still detected
        v1 = self.make_v1(self.z['Z']['frozen'])
        cid = sorted(v1['predictions'])[0]
        v1['predictions'][cid][0]['C3'] += 1e-6
        p = self.tmp / 'frozen_v1_bad.json'
        p.write_text(json.dumps(self.rehash(v1)))
        r = self.verify(p, params=self.z['Z']['params'])
        self.assertEqual(r.returncode, 2)
        self.assertIn('predictions recompute', r.stdout)

    def test_tampered_c3z_prediction_detected(self):
        doc = copy.deepcopy(self.z['Z']['frozen'])
        cid = sorted(doc['predictions'])[0]
        doc['predictions'][cid][0]['C3Z'] += 1e-6
        p = self.tmp / 'frozen_c3z_edit.json'
        p.write_text(json.dumps(doc))
        r = self.verify(p, params=self.z['Z']['params'])                 # hash no longer matches contents
        self.assertEqual(r.returncode, 2)
        self.assertIn('self-hash', r.stdout)
        p.write_text(json.dumps(self.rehash(doc)))                       # forged hash: recomputation catches it
        r = self.verify(p, params=self.z['Z']['params'])
        self.assertEqual(r.returncode, 2)
        self.assertIn('predictions recompute', r.stdout)
        r = self.evaluate(p, self.build, self.z['Z']['params'], self.tmp / 'counts_Z_C3Z.json', self.tmp / 'x.json')
        self.assertEqual(r.returncode, 2)
        self.assertIn('REFUSED', r.stderr)
        # removing the C3Z field from a v2 file is also detected (the hash covers it)
        doc = copy.deepcopy(self.z['Z']['frozen'])
        del doc['predictions'][cid][0]['C3Z']
        p.write_text(json.dumps(doc))
        self.assertEqual(self.verify(p, params=self.z['Z']['params']).returncode, 2)
        # changing alpha in the parameter file is detected
        bad = copy.deepcopy(PARAMS_Z)
        bad['cz']['16_23']['alpha']['23'] += 1e-6
        pf = self.tmp / 'params_Z_bad.json'
        write_lf(pf, json.dumps(bad) + '\n')
        r = self.verify(self.z['Z']['frozen_path'], params=pf)
        self.assertEqual(r.returncode, 2)
        self.assertIn('parameter file', r.stdout)

    def test_original_dryrun_behaviour_unchanged(self):
        cands = self.reports['C3']['analysis']['candidates']
        self.assertEqual(list(cands), ['C3', 'C2', 'C3Z'])               # primary, secondary, then the extension last
        self.assertEqual(cands['C3']['verdict'], 'PASS')
        self.assertEqual(cands['C3']['numerical_vs_physical']['models_considered'], 'all four models (worst case)')
        self.assertIn('=== C3 (PRIMARY)', self.reports['C3_text'])
        self.assertEqual(self.reports['C0']['analysis']['candidates']['C3']['verdict'], 'FAIL')


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
