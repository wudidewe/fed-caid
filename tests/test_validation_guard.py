"""Secure arithmetic, partition coverage and artifact boundaries."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ids_core as core
import privacy_run as run
from secure_aggregation import SecureAggregationSession
from validation_core import privacy_scope_validation, validate_aggregate, validate_role_coverage


class ValidationGuardTests(unittest.TestCase):
    def test_linear_partition_coverage_detects_internal_and_cross_duplicates(self):
        self.assertTrue(validate_role_coverage([np.array([2, 0]), np.array([1, 3])], 4))
        for groups in (([0, 0], [1, 2]), ([0, 1], [1, 2]), ([0], [1, 2]), ([0, 1], [2, 4])):
            with self.assertRaises(ValueError):
                validate_role_coverage([np.array(x) for x in groups], 4)

    def test_secure_mean_matches_private_reference_and_detects_corruption(self):
        updates = np.array([[.01234567, -.00432123], [-.00987654, .00432123], [.00111111, .00222222]])
        with SecureAggregationSession(['a', 'b', 'c'], 2, clip=.1, scale=1000000) as channel:
            contract = channel.start_round(session='test', round_index=1, parent_model='0' * 64)
            for client, delta in zip(('a', 'b', 'c'), updates):
                channel.submit(client, delta, contract=contract)
            recovered, _ = channel.aggregate()
        result = validate_aggregate(updates, recovered, 1000000)
        self.assertTrue(result['passed'])
        self.assertLessEqual(result['max_coordinate_error'], .5e-6)
        with self.assertRaises(RuntimeError): validate_aggregate(updates, recovered + 1e-4, 1000000)

    def test_public_scope_rejects_individual_artifacts_and_false_DP(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT) as temp:
            public = Path(temp)
            core.save_json(public / 'protocol.json', {'differential_privacy': False, 'epsilon': None, 'delta': None})
            self.assertEqual(privacy_scope_validation(public)['public_individual_artifact_scan'], 'pass')
            for leak in ({'nested': [{'log_prior': [0., 1.]}]}, {'shares': [1, 2]}):
                core.save_json(public / 'bad.json', leak)
                with self.assertRaises(RuntimeError): privacy_scope_validation(public)
            (public / 'bad.json').unlink()
            (public / 'model.pt').write_bytes(b'fixture')
            with self.assertRaises(RuntimeError): privacy_scope_validation(public)
            (public / 'model.pt').unlink()
            core.save_json(public / 'protocol.json', {'differential_privacy': True, 'epsilon': 1., 'delta': 1e-5})
            with self.assertRaises(RuntimeError): privacy_scope_validation(public)


if __name__ == "__main__":
    unittest.main()
