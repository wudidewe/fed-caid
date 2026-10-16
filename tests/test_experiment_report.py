"""Presentation boundaries that protect inputs and scientific interpretation."""
from pathlib import Path
from copy import deepcopy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment_report import _communication_totals, _prediction_measurement, _stat_text, _plots, generate_report, main
from experiment_report_data import ReportDataError, load_runs
from test_experiment_report_data_loader import fixture


class ReportPresentationTests(unittest.TestCase):
    def test_complete_report_exports_files_and_loads_observation_dependency(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory)
            fixture(root/'run')
            report=generate_report([root/'run'],root/'report')
            self.assertTrue(report['complete'])
            self.assertTrue(report['input_json_files_unchanged']>0)
            self.assertTrue((root/'report/report.html').is_file())
            self.assertTrue((root/'report/metrics_summary.csv').is_file())
            self.assertTrue((root/'report/manifest.json').is_file())
            for name in report['figures']:
                graphic=root/'report'/'figures'/(name+'.png')
                self.assertTrue(graphic.is_file())
                self.assertTrue(graphic.with_suffix('.pdf').is_file())

    def test_missing_byte_counter_is_unknown_instead_of_zero(self):
        self.assertIsNone(_communication_totals([{'communication': {}}], 'local_ipc_frame_bytes'))
        self.assertEqual(_communication_totals([{'communication': {'local_ipc_frame_bytes': 0}}], 'local_ipc_frame_bytes'), 0)
        self.assertEqual(_communication_totals([{'communication': {'numeric_share_upload_bytes': 120}},
                                               {'communication': {'numeric_share_upload_bytes': 240}}],
                                              'numeric_share_upload_bytes'), 360)

    def test_single_run_format_contains_only_its_observed_value(self):
        self.assertEqual(_stat_text({'mean': .8, 'sample_sd': None}), '80.000')

    def test_cli_rejects_multiple_inputs_before_report_generation(self):
        with patch('experiment_report.generate_report') as generate, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['--runs', 'first_run', 'second_run'])
        self.assertEqual(error.exception.code, 2)
        generate.assert_not_called()

    def test_standalone_figures_keep_equal_matrices_and_independent_scopes(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            fixture(root / 'run')
            data = load_runs([root / 'run'])
            original = data['summary']['D']
            names = ['CIC-IoV-2024', 'Car-Hacking', 'CIC-IDS-2017', 'CIC-UNSW-NB15']
            data['datasets'] = names
            data['summary'] = {name: deepcopy(original) for name in names}
            data['runs'][0]['datasets'] = {name: deepcopy(data['runs'][0]['datasets']['D']) for name in names}
            for name in names:
                data['summary'][name]['aggregation'][0]['rounds'][0]['communication'] = {
                    'numeric_share_upload_bytes': 120, 'local_ipc_frame_bytes': 500}
            data['summary'][names[2]]['confusion']['matrix'] = [[0, 10], [1, 1999]]
            checked, matrix_sizes = [], []
            def inspect(fig, folder, name, figures):
                fig.canvas.draw()
                self.assertIsNone(fig._suptitle)
                if name.startswith('confusion_'):
                    self.assertEqual(len(fig.axes), 2)
                    axis, colorbar = fig.axes
                    self.assertEqual(colorbar.get_ylabel(), 'Row percent')
                    self.assertAlmostEqual(axis.get_position().height, colorbar.get_position().height, places=7)
                    self.assertEqual(fig.get_size_inches().tolist(), [8., 7.4])
                    matrix_sizes.append((axis.get_position().width * 8, axis.get_position().height * 7.4))
                    self.assertFalse(any('n=' in tick.get_text() for tick in axis.get_yticklabels()))
                    if name == 'confusion_cicids2017':
                        self.assertEqual([text.get_text() for text in axis.texts], ['0', '100', '<0.1', '100'])
                else:
                    self.assertEqual(len(fig.axes), 1)
                    if name == 'communication_volume':
                        self.assertEqual([bar.get_height() for bar in fig.axes[0].patches], [120] * 4 + [500] * 4)
                checked.append(name)
                figures.append(name)
            output = root / 'paper'
            output.mkdir()
            with patch('experiment_report._save', side_effect=inspect):
                _plots(data, output)
            self.assertEqual(checked, ['confusion_cicids2017', 'confusion_cicunsw_nb15',
                                       'training_time', 'prediction_time', 'communication_volume'])
            self.assertEqual(matrix_sizes[0], matrix_sizes[1])
            for width, height in matrix_sizes:
                self.assertAlmostEqual(width, 5.6)
                self.assertAlmostEqual(height, 5.6)
            data['datasets'] = names[:2]
            output = root / 'can'
            output.mkdir()
            with patch('experiment_report._save', side_effect=lambda fig, folder, name, figures: figures.append(name)):
                figures = _plots(data, output)
            self.assertEqual(figures, ['training_time', 'prediction_time', 'communication_volume'])
        plt.close('all')

    def test_prediction_uses_ratio_of_client_totals_and_validates_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            run = Path(directory) / 'run'
            fixture(run)
            path = run / 'private_DO_NOT_UPLOAD/D/performance_private.json'
            raw = json.loads(path.read_text())
            for client, seconds, rows in zip(raw['clients'], (.1, .2, .3), (100, 200, 700)):
                client.update(inference_seconds=seconds, inference_rows=rows,
                              inference_ms_per_sample=1000 * seconds / rows)
            path.write_text(json.dumps(raw), encoding='utf-8')
            measured = _prediction_measurement(load_runs([run]), 'D')
            self.assertAlmostEqual(measured['seconds_sum_clients'], .6)
            self.assertEqual(measured['rows_sum_clients'], 1000)
            self.assertAlmostEqual(measured['microseconds_per_record'], 600.)
            # An arithmetic average of three client rates would be wrong.
            self.assertNotAlmostEqual(measured['microseconds_per_record'], (1000 + 1000 + 300000 / 700) / 3)
            for invalid in (0, -1, 1.5, True):
                with self.subTest(rows=invalid):
                    candidate = deepcopy(raw)
                    candidate['clients'][0]['inference_rows'] = invalid
                    path.write_text(json.dumps(candidate), encoding='utf-8')
                    with self.assertRaisesRegex(ReportDataError, 'inference_rows'):
                        load_runs([run])
            candidate = deepcopy(raw)
            candidate['clients'][0]['inference_ms_per_sample'] = 2.
            path.write_text(json.dumps(candidate), encoding='utf-8')
            with self.assertRaisesRegex(ReportDataError, 'disagrees'):
                load_runs([run])

    def test_missing_prediction_denominator_or_timer_is_never_imputed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            run = Path(directory) / 'run'
            fixture(run)
            path = run / 'private_DO_NOT_UPLOAD/D/performance_private.json'
            raw = json.loads(path.read_text())
            for client in raw['clients']:
                client['inference_seconds'] = .1
                client['inference_rows'] = 10
            for key in ('inference_rows', 'inference_seconds'):
                candidate = deepcopy(raw)
                candidate['clients'][0].pop(key)
                path.write_text(json.dumps(candidate), encoding='utf-8')
                measured = _prediction_measurement(load_runs([run]), 'D')
                self.assertIsNone(measured['microseconds_per_record'])

    def test_original_run_tree_and_existing_report_cannot_be_written(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            run = root / 'original_run'
            run.mkdir()
            with self.assertRaisesRegex(ReportDataError, 'outside'):
                generate_report([run], run / 'report')
            with self.assertRaisesRegex(ReportDataError, 'NEW'):
                generate_report([run], root)
            self.assertEqual(list(run.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
