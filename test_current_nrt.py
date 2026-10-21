"""Guard the migration from legacy NRT to regularization and saved cache reuse."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import workflow
from regularization import workflow as current
from tuning.main import parser


class CurrentNRTTests(unittest.TestCase):
    def tearDown(self):
        current.source_hash.cache_clear()

    def test_benchmark_dispatch_excludes_retired_methods(self):
        self.assertEqual(set(workflow.METHODS), set(workflow.BENCHMARK_BACKENDS))
        for method in ('NRT_NLL', 'NRT_Brier', 'NRT_2Loss'):
            with self.assertRaises(ValueError):
                workflow.select_methods(method)
        method_action = next(a for a in parser()._actions if a.dest == 'method')
        self.assertEqual(set(method_action.choices), {'NNCDE', 'DeepSurv'})

    def test_current_source_uses_alias_only_for_exact_match(self):
        mapping = json.loads((workflow.ROOT / 'environment/source-compatibility.json').read_text())
        digest = hashlib.sha256()
        for name in mapping['files']:
            digest.update(name.encode())
            digest.update((workflow.ROOT / name).read_bytes())
        current.source_hash.cache_clear()
        raw = digest.hexdigest()
        self.assertEqual(current.source_hash(), mapping['aliases'].get(raw, raw))

    def test_historical_cleanup_alias_is_still_honored(self):
        mapping = json.loads((workflow.ROOT / 'environment/source-compatibility.json').read_text())
        for raw, saved in mapping['aliases'].items():
            with self.subTest(raw=raw), patch.object(current.hashlib, 'sha256') as digest:
                digest.return_value.hexdigest.return_value = raw
                current.source_hash.cache_clear()
                self.assertEqual(current.source_hash(), saved)

    def test_numerical_source_change_does_not_reuse_old_fingerprint(self):
        current.source_hash.cache_clear()
        before = current.source_hash()
        original = Path.read_bytes
        def changed(path):
            data = original(path)
            return data + b'\n# changed model source\n' if path == current.ROOT / 'model.py' else data
        current.source_hash.cache_clear()
        with patch.object(Path, 'read_bytes', changed):
            self.assertNotEqual(current.source_hash(), before)
