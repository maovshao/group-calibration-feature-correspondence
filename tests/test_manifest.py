"""Offline orchestration tests: all records are synthetic and temporary."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_summarize import synthetic_run, summary

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('runner_under_test', ROOT / 'code/run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='calibration-test-', dir='/tmp')
        self.root = Path(self.temp.name)
        (self.root / 'code').mkdir()
        for name in ('run.py', 'preflight.py'):
            shutil.copyfile(ROOT / 'code' / name, self.root / 'code' / name)
        shutil.copytree(ROOT / 'code/configs', self.root / 'code/configs')
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def invoke(self, argv, effect):
        with patch.object(sys, 'argv', ['run.py', *argv]), patch.object(runner, 'execute', side_effect=effect) as execute:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = runner.main()
        return result, execute

    @staticmethod
    def completed(metadata, directory, plan, observer):
        metadata['status'] = 'completed'

    def test_manifest_lists_exact_batch_and_no_previous_attempts(self):
        old = self.root / 'outputs/runs/old/metadata.json'
        old.parent.mkdir(parents=True)
        old.write_text('{}')
        result, execute = self.invoke(['--level', 'main', '--method', 'all', '--seeds', '0', '1', '2', '3', '4',
                                     '--manifest-out', 'outputs/main.json'], self.completed)
        self.assertEqual(result, 0)
        self.assertEqual(execute.call_count, 20)
        paths = json.loads((self.root / 'outputs/main.json').read_text())
        self.assertEqual(len(paths), 20)
        self.assertEqual(len(set(paths)), 20)
        records = [json.loads((self.root / p).read_text()) for p in paths]
        self.assertEqual({(r['method'], r['seed']) for r in records},
                         {(m, s) for m in runner.METHODS for s in range(5)})
        self.assertNotIn(str(old.relative_to(self.root)), paths)
        self.assertTrue(all(not Path(p).is_absolute() for p in paths))

    def test_existing_manifest_rejected_before_any_run(self):
        destination = self.root / 'batch.json'
        destination.write_text('["preserve"]')
        with patch.object(runner, 'execute') as execute:
            with self.assertRaises(SystemExit) as error:
                self.invoke(['--manifest-out', str(destination)], self.completed)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(destination.read_text(), '["preserve"]')
        self.assertFalse((self.root / 'outputs/runs').exists())
        execute.assert_not_called()

    def test_failed_attempt_stays_in_manifest_and_has_exit_status(self):
        result, _ = self.invoke(['--manifest-out', 'batch.json'], RuntimeError('synthetic failure'))
        self.assertEqual(result, 1)
        paths = json.loads((self.root / 'batch.json').read_text())
        self.assertEqual(len(paths), 1)
        record = json.loads((self.root / paths[0]).read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['exit_code'], 1)
        self.assertEqual((self.root / paths[0]).with_name('exit_code.txt').read_text(), '1\n')

    def test_blocked_prerequisites_stop_remaining_batch(self):
        result, execute = self.invoke(['--level', 'main', '--method', 'all', '--seeds', '0', '1',
                                      '--manifest-out', 'batch.json'], runner.Blocked('synthetic blocker'))
        self.assertEqual(result, 2)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(json.loads((self.root / 'batch.json').read_text())), 1)

    def test_duplicate_seeds_rejected_before_reserving_manifest(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(['--level', 'main', '--seeds', '0', '0', '--manifest-out', 'batch.json'], self.completed)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / 'batch.json').exists())

    def test_manifest_rejects_duplicate_normalized_paths(self):
        manifest = self.root / 'batch.json'
        manifest.write_text(json.dumps(['outputs/a.json', './outputs/a.json']))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summary.load_manifest_paths(manifest, self.root)

    def test_manifest_rejects_non_list_and_non_path_items(self):
        manifest = self.root / 'batch.json'
        for value in ({'runs': []}, [''], [None], [3]):
            manifest.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                summary.load_manifest_paths(manifest, self.root)

    def summarize_batch(self, records):
        selected = []
        for i, record in enumerate(records):
            path = self.root / f'fixture{i}.json'
            path.write_text(json.dumps(record))
            selected.append(str(path))
        manifest = self.root / 'selected.json'
        manifest.write_text(json.dumps(selected))
        return subprocess.run([sys.executable, str(ROOT / 'code/summarize.py'),
                               '--manifest', str(manifest), '--output', str(self.root / 'summary')],
                              text=True, capture_output=True)

    def test_incomplete_main_batch_rejected_by_default(self):
        result = self.summarize_batch([synthetic_run()])
        self.assertEqual(result.returncode, 1, result.stderr)
        status = json.loads((self.root / 'summary/summary_status.json').read_text())
        self.assertEqual(status['status'], 'partial')
        self.assertFalse(status['numeric_outputs_written'])
        self.assertEqual(list((self.root / 'summary').glob('*.csv')), [])

    def test_complete_main_batch_accepted_by_default(self):
        result = self.summarize_batch([synthetic_run(m, s) for m in runner.METHODS for s in range(5)])
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads((self.root / 'summary/summary_status.json').read_text())
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(list((self.root / 'summary').glob('*.csv'))), 4)

    def test_failed_extra_attempt_invalidates_selection(self):
        records = [synthetic_run(m, s) for m in runner.METHODS for s in range(5)]
        records.append(synthetic_run(status='failed', run_id='FAILED_FIXTURE'))
        result = self.summarize_batch(records)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.root / 'summary/summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
