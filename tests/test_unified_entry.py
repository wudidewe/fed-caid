"""Public dataset selection, isolated profiles, and report matching contracts."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from dataset_access import DATASET_PROFILES


class UnifiedEntryTests(unittest.TestCase):
    def test_all_four_profiles_keep_previous_parameters_and_independent_dicts(self):
        args=main.parser().parse_args([])
        self.assertEqual(main.selected_datasets(args),main.DATASETS)
        self.assertEqual(set(main.DATASETS),set(DATASET_PROFILES))
        for name,features,classes in zip(main.DATASETS,(9,10,68,68),(6,5,15,10)):
            resolved=main.resolve_dataset_parameters(args,name)
            self.assertEqual((resolved.base_epochs,resolved.batch_size,resolved.clients,resolved.rounds),(5,1024,3,3))
            self.assertEqual((resolved.detector_lr,resolved.detector_weight_decay),(.002,.0001))
            self.assertEqual(len(DATASET_PROFILES[name]['feature_names']),features)
            self.assertEqual(len(DATASET_PROFILES[name]['class_names']),classes)
        self.assertEqual(len({id(value) for value in main.DATASET_PARAMETERS.values()}),4)

    def test_profile_resolution_does_not_leak_or_change_cli_override(self):
        args=main.parser().parse_args(['--batch-size','7'])
        args._explicit_options={'batch_size'}
        first=main.DATASETS[0]
        with patch.dict(main.DATASET_PARAMETERS[first],{'base_epochs':2,'batch_size':11}):
            a=main.resolve_dataset_parameters(args,first)
            b=main.resolve_dataset_parameters(args,main.DATASETS[1])
        self.assertEqual((a.base_epochs,a.batch_size),(2,7))
        self.assertEqual((b.base_epochs,b.batch_size),(5,7))
        self.assertEqual((args.base_epochs,args.batch_size),(5,7))

    def test_single_dataset_and_scope_conflicts_are_explicit(self):
        args=main.parser().parse_args(['--dataset','CIC-IDS-2017'])
        self.assertEqual(main.selected_datasets(args),('CIC-IDS-2017',))
        args.scope='can'
        with self.assertRaises(ValueError): main.selected_datasets(args)
        args=main.parser().parse_args(['--dataset','auto'])
        with self.assertRaises(ValueError): main.selected_datasets(args)

    def test_default_entry_prepares_selected_repository_datasets_before_training(self):
        with patch.object(main,'prepare_selected_datasets') as prepare, patch('privacy_run.execute',return_value=9) as execute:
            self.assertEqual(main.main([]),9)
        self.assertEqual(main.selected_datasets(execute.call_args.args[0]),main.DATASETS)
        prepare.assert_called_once_with(execute.call_args.args[0],main.DATASETS)
        self.assertEqual(Path(execute.call_args.args[0].data_root),main.ROOT/'dataset')

    def test_invalid_per_dataset_budget_is_rejected_before_training(self):
        args=main.parser().parse_args([])
        args._explicit_options=set()
        with patch.dict(main.DATASET_PARAMETERS[main.DATASETS[0]],{'clients':2}):
            with self.assertRaises(ValueError): main.resolve_dataset_parameters(args,main.DATASETS[0])


if __name__=='__main__': unittest.main()
