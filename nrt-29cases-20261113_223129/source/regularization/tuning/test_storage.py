"""Compact-output safety and interrupted-search recovery checks."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .main import (SearchConfig, compact_case_outputs, score_candidate, search_method)
from ..unified_nll.nll_cv_train import NLLCVOptimization
from ..unified_nll.train import TrainingConfig
from ..unified_nll.tuning_model import Architecture
from ..workflow import MATCHED, METHODS


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        for method in METHODS:
            final = self.root/method/'case'
            final.mkdir(parents=True)
            (final/'selection.json').write_text(json.dumps(dict(
                parameters=dict(depth=3), scores=dict(cv_score=1., trial=-1), request_hash='same')))
            (final/'result.json').write_text(json.dumps(dict(status='ok', request_hash='same')))
            (final/'history.json').write_text('[]')
            work = self.root/'studies/case'/method
            cache = work/'fold_cache/trial_0'
            cache.mkdir(parents=True)
            (cache/'fold_1.json').write_text('{}')
            (cache/'history_1.json').write_text('[]')
            (work/'study.sqlite3').write_bytes(b'placeholder')
            if method != MATCHED:
                (work/'cv_results.csv').write_text('trial,cv_score\n0,1.0\n')
        (self.root/'studies/case/request.json').write_text('{}')
        (self.root/'studies/case/folds.json').write_text('{}')

    def test_compaction_preserves_final_outputs_and_is_idempotent(self):
        self.fixture()
        originals = {p: p.read_bytes() for m in METHODS for p in (self.root/m/'case').glob('*.json')}
        for _ in range(2):
            self.assertTrue(compact_case_outputs(self.root, 'case', METHODS))
            for method in METHODS:
                self.assertEqual({p.name for p in (self.root/method/'case').iterdir()},
                                 {'cv_results.csv', 'selection.json', 'result.json', 'history.json'})
            self.assertFalse(list((self.root/'studies').rglob('*.json')))
            self.assertFalse(list((self.root/'studies').rglob('*.sqlite3')))
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_incomplete_outputs_leave_caches_untouched(self):
        self.fixture()
        (self.root/MATCHED/'case/history.json').unlink()
        self.assertFalse(compact_case_outputs(self.root, 'case', METHODS))
        self.assertEqual(len(list(self.root.rglob('fold_1.json'))), 3)

    def test_missing_cv_table_prevents_cleanup(self):
        self.fixture()
        (self.root/'studies/case'/METHODS[0]/'cv_results.csv').unlink()
        with self.assertRaisesRegex(ValueError, 'missing CV results'):
            compact_case_outputs(self.root, 'case', METHODS)
        self.assertEqual(len(list(self.root.rglob('fold_1.json'))), 3)

    def test_folds_are_aggregated_without_file_writes(self):
        data = SimpleNamespace(bounds=lambda indices: indices)
        infos = [dict(best_epoch=2, validation_nll=1.), dict(best_epoch=5, validation_nll=3.)]
        with patch('regularization.tuning.main.fit_raw_time_kl', side_effect=[
                (SimpleNamespace(fit_info=info), [dict(epoch=1)]) for info in infos]), \
             patch('regularization.tuning.main.atomic_json') as save_json:
            scores = score_candidate(data, [([0], [1]), ([1], [0, 2, 3])], Architecture(),
                TrainingConfig(), NLLCVOptimization(), SearchConfig(), 'cpu', upper_bound=1.)
        save_json.assert_not_called()
        self.assertEqual(scores, dict(cv_score=2.5, cv_best_epochs=[2, 5], refit_epochs=4))

    def test_search_resumes_completed_trials_without_fold_files(self):
        folder = self.root/'study'
        cv_path = self.root/'final/cv_results.csv'
        data = SimpleNamespace(case=SimpleNamespace(index=0, id='case'))
        search = SearchConfig(trials=2, initial_trials=2)
        args = (folder, data, [], METHODS[0], search, Architecture(),
                TrainingConfig(), NLLCVOptimization(), 'cpu')
        scores = dict(cv_score=1., cv_best_epochs=[2, 2], refit_epochs=2)
        with patch('regularization.tuning.main.score_candidate',
                   side_effect=[scores, RuntimeError('interrupted')]):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                search_method(*args, upper_bound=1., cv_path=cv_path)
        with cv_path.open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)
        with patch('regularization.tuning.main.score_candidate', return_value=scores) as fit:
            search_method(*args, upper_bound=1., cv_path=cv_path)
            self.assertEqual(fit.call_count, 1)
        with cv_path.open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        self.assertFalse(list(self.root.rglob('fold*')))


if __name__ == '__main__':
    unittest.main()
