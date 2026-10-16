"""Pinned raw-source downloads with small in-memory synthetic CSV fixtures."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import kaggle_source as source


class KaggleSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = 'CIC-UNSW-NB15'
        self.payload = b'value,label\n1,BENIGN\n2,Attack\n'
        self.recipe = {'name': self.dataset,
                       'kaggle': {'handle': source.KAGGLE_HANDLES[self.dataset], 'version': 1},
                       'source_files': [self.file('CICFlowMeter_out.csv', self.payload,
                                                  kaggle_path='CICFlowMeter.csv')]}

    def file(self, name, payload, **extra):
        return {'filename': name, 'bytes': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest(), **extra}

    def download(self, handle, name, output):
        output.mkdir(parents=True)
        path = output / Path(name).name
        path.write_bytes(self.payload)
        return path

    def run_ensure(self, recipe=None):
        with redirect_stdout(io.StringIO()), patch.object(source, '_temporary_parent', return_value=self.root):
            return source.ensure_kaggle_sources(self.dataset, self.root, recipe or self.recipe)

    def test_pinned_version_and_explicit_renaming_preserve_file_content(self):
        self.recipe['source_files'][0]['path'] = str(self.root.parent / 'ignored.csv')
        with patch.object(source, '_download_file', side_effect=self.download) as fetch:
            self.assertIsNone(self.run_ensure())
        handle, remote, _ = fetch.call_args.args
        self.assertEqual(handle, source.KAGGLE_HANDLES[self.dataset] + '/versions/1')
        self.assertEqual(remote, 'CICFlowMeter.csv')
        self.assertEqual((self.root / 'raw' / self.dataset / 'CICFlowMeter_out.csv').read_bytes(), self.payload)
        self.assertFalse(list(self.root.glob('fedcaid_*')))

    def test_complete_verified_raw_directory_skips_network(self):
        raw = self.root / 'raw' / self.dataset
        raw.mkdir(parents=True)
        (raw / 'CICFlowMeter_out.csv').write_bytes(self.payload)
        with patch.object(source, '_download_file', side_effect=AssertionError('No download expected')):
            self.run_ensure()

    def test_iov_archive_directory_can_be_flat_under_dataset_root(self):
        self.dataset = 'CIC-IoV-2024'
        recipe = {'name': self.dataset, 'raw_directory': 'archive',
                  'kaggle': {'handle': source.KAGGLE_HANDLES[self.dataset], 'version': 1},
                  'source_files': [self.file('decimal_benign.csv', self.payload)]}
        with patch.object(source, '_download_file', side_effect=self.download):
            self.run_ensure(recipe)
        self.assertEqual((self.root / 'archive/decimal_benign.csv').read_bytes(), self.payload)
        self.assertFalse((self.root / 'raw').exists())

    def test_bad_existing_file_is_not_overwritten(self):
        raw = self.root / 'raw' / self.dataset
        raw.mkdir(parents=True)
        target = raw / 'CICFlowMeter_out.csv'
        target.write_bytes(b'wrong')
        with patch.object(source, '_download_file', side_effect=AssertionError('No download expected')):
            with self.assertRaisesRegex(ValueError, 'byte count'):
                self.run_ensure()
        self.assertEqual(target.read_bytes(), b'wrong')

    def test_partial_raw_directory_fetches_only_its_missing_required_file(self):
        raw = self.root / 'raw' / self.dataset
        raw.mkdir(parents=True)
        first = raw / 'CICFlowMeter_out.csv'
        first.write_bytes(self.payload)
        before = first.stat().st_mtime_ns
        self.recipe['source_files'].append(self.file('second.csv', self.payload))
        with patch.object(source, '_download_file', side_effect=self.download) as fetch:
            self.run_ensure()
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[1], 'second.csv')
        self.assertEqual(first.stat().st_mtime_ns, before)
        self.assertEqual((raw / 'second.csv').read_bytes(), self.payload)

    def test_all_required_files_are_verified_before_any_installation(self):
        self.recipe['source_files'].append(self.file('second.csv', b'not-the-downloaded-file'))
        with patch.object(source, '_download_file', side_effect=self.download):
            with self.assertRaises(ValueError):
                self.run_ensure()
        self.assertFalse((self.root / 'raw' / self.dataset).exists())
        self.assertFalse(list(self.root.glob('fedcaid_*')))

    def test_same_size_corruption_is_rejected_by_sha(self):
        recipe = copy.deepcopy(self.recipe)
        recipe['source_files'][0]['sha256'] = '0' * 64
        with patch.object(source, '_download_file', side_effect=self.download):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.run_ensure(recipe)

    def test_single_csv_zip_is_unpacked_without_changing_the_original_bytes(self):
        def zipped(handle, filename, output):
            output.mkdir(parents=True)
            saved = output / 'CICFlowMeter.csv'
            with zipfile.ZipFile(saved, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(filename, self.payload)
            return saved
        with patch.object(source, '_download_file', side_effect=zipped):
            self.run_ensure()
        self.assertEqual((self.root / 'raw' / self.dataset / 'CICFlowMeter_out.csv').read_bytes(), self.payload)

    def test_single_csv_zip_rejects_extra_members_paths_and_wrong_size(self):
        for entries in ([('wrong.csv', self.payload)], [('../CICFlowMeter.csv', self.payload)],
                        [('CICFlowMeter.csv', b'short')],
                        [('CICFlowMeter.csv', self.payload), ('other.csv', self.payload)]):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory(dir=self.root) as folder:
                saved = Path(folder) / 'download'
                with zipfile.ZipFile(saved, 'w') as archive:
                    for name, content in entries:
                        archive.writestr(name, content)
                with self.assertRaises(ValueError):
                    source._unpack_csv(saved, 'CICFlowMeter.csv', len(self.payload))

    def test_missing_or_invalid_pinned_version_cannot_use_latest(self):
        for version in (None, 0, -1, True, '1'):
            recipe = copy.deepcopy(self.recipe)
            recipe['kaggle']['version'] = version
            with self.subTest(version=version), patch.object(source, '_download_file') as fetch:
                with self.assertRaises(ValueError):
                    self.run_ensure(recipe)
                fetch.assert_not_called()

    def test_unsupported_source_or_dataset_is_rejected_before_network(self):
        recipe = copy.deepcopy(self.recipe)
        recipe['kaggle']['handle'] = 'other/unverified-source'
        with patch.object(source, '_download_file') as fetch:
            with self.assertRaises(ValueError):
                self.run_ensure(recipe)
            fetch.assert_not_called()

    def test_path_traversal_and_duplicate_names_are_rejected(self):
        for field, bad in (('filename', '../outside.csv'), ('filename', 'a/b.csv'),
                           ('filename', 'C:\\outside.csv'), ('kaggle_path', '../outside.csv')):
            recipe = copy.deepcopy(self.recipe)
            recipe['source_files'][0][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                self.run_ensure(recipe)
        recipe = copy.deepcopy(self.recipe)
        recipe['source_files'].append(copy.deepcopy(recipe['source_files'][0]))
        with self.assertRaises(ValueError):
            self.run_ensure(recipe)
        for bad in ('../outside', '/absolute', 'C:/absolute', 'raw\\dataset'):
            recipe = copy.deepcopy(self.recipe)
            recipe['raw_directory'] = bad
            with self.subTest(raw_directory=bad), self.assertRaises(ValueError):
                self.run_ensure(recipe)

    def test_transport_calls_official_api_with_version_and_local_output(self):
        output = self.root / 'staging'
        output.mkdir()
        target = output / 'CICFlowMeter.csv'
        target.write_bytes(self.payload)
        api = types.SimpleNamespace(dataset_download=lambda *args, **kwargs: str(target))
        with patch.object(source.importlib, 'import_module', return_value=api), \
                patch.object(api, 'dataset_download', return_value=str(target)) as fetch:
            actual = source._download_file('owner/name/versions/1', 'CICFlowMeter.csv', output)
        self.assertEqual(actual, target)
        fetch.assert_called_once_with('owner/name/versions/1', path='CICFlowMeter.csv', output_dir=str(output))

    def test_transport_rejects_unexpected_return_location_and_failed_download(self):
        api = types.SimpleNamespace(dataset_download=lambda *args, **kwargs: str(self.root / 'outside.csv'))
        with patch.object(source.importlib, 'import_module', return_value=api):
            with self.assertRaisesRegex(RuntimeError, 'outside'):
                source._download_file('owner/name/versions/1', 'name.csv', self.root / 'stage')
        with patch.object(source.importlib, 'import_module', side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, 'require kagglehub'):
                source._download_file('owner/name/versions/1', 'name.csv', self.root / 'stage')
        with patch.object(source.importlib, 'import_module', return_value=api), \
                patch.object(api, 'dataset_download', side_effect=ConnectionError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Kaggle download failed'):
                source._download_file('owner/name/versions/1', 'name.csv', self.root / 'stage')


if __name__ == '__main__':
    unittest.main()
