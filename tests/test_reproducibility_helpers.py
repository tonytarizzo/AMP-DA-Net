import unittest
import tempfile
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.decoding import paper_count_projection
from src.paths import _fmt_float, load_raw_ifed_bundle
from src.paths.pretrain_handling import find_decoder_artifacts

corrupt_deltas = importlib.import_module("scripts.3_feel_inference").corrupt_deltas
aggregate_sparse_vectors = importlib.import_module("scripts.3_feel_inference").aggregate_sparse_vectors
pretraining_module = importlib.import_module("scripts.2_pretraining")
load_raw_splits = pretraining_module.load_raw_splits


class FloatFormattingTests(unittest.TestCase):
    def test_integer_zeroes_are_not_stripped(self):
        self.assertEqual(_fmt_float(20.0, 0), "20")
        self.assertEqual(_fmt_float(0.0, 0), "0")

    def test_fractional_values_remain_compact(self):
        self.assertEqual(_fmt_float(0.20, 2), "0.2")
        self.assertEqual(_fmt_float(3.50, 2), "3.5")


class PaperProjectionTests(unittest.TestCase):
    def test_projection_is_nonnegative_integer_and_sum_k(self):
        projected = paper_count_projection(np.array([-2.0, 0.2, 1.7, 0.9]), 3)
        self.assertTrue(np.issubdtype(projected.dtype, np.integer))
        self.assertTrue(np.all(projected >= 0))
        self.assertEqual(int(projected.sum()), 3)

    def test_projection_handles_large_single_coordinate(self):
        projected = paper_count_projection(np.array([10.2, 0.1]), 1)
        np.testing.assert_array_equal(projected, np.array([1, 0]))


class DatasetMetadataWarningTests(unittest.TestCase):
    def test_device_count_mismatch_uses_argparse_attribute_names(self):
        args = SimpleNamespace(message_split=20, K_t=40, min_p=7, max_p=13)
        bundle = {
            "L": 20,
            "meta": {"kind": "ifed_raw_per_device", "K_t": 20, "min_p": 7, "max_p": 13},
        }
        with patch("src.paths.dataset_handling.auto_find_raw_dataset", return_value="bundle.pth"), \
             patch("src.paths.dataset_handling.torch.load", return_value=bundle), \
             patch("src.paths.dataset_handling._model_tag_from_args", return_value="resnet"), \
             patch("builtins.print") as mocked_print:
            load_raw_ifed_bundle(args, "cpu")
        messages = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("K_t differs: dataset=20 vs args=40", messages)


class LegacyCheckpointTests(unittest.TestCase):
    def test_corrected_slug_can_load_legacy_range_name(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_base = Path(directory) / "run_snrU[-2]dB"
            legacy_dec = Path(str(legacy_base) + ".pth")
            legacy_cb = Path(str(legacy_base) + "_cb.pt")
            legacy_dec.touch()
            legacy_cb.touch()
            args = SimpleNamespace(
                decoder_path="", codebook_path="", decoder_slug="",
                ckpt_dir=directory, snr_mode="range", snr_min_db=0.0,
                snr_max_db=20.0,
            )
            with patch("src.paths.pretrain_handling.make_decoder_slug", return_value="run_snrU[0-20]dB"):
                decoder, codebook = find_decoder_artifacts(args)
            self.assertEqual(Path(decoder), legacy_dec)
            self.assertEqual(Path(codebook), legacy_cb)


class CorruptionSmokeTests(unittest.TestCase):
    def test_seeded_corruption_path_is_supported(self):
        import torch

        deltas = torch.ones((3, 8), dtype=torch.float32)
        first = corrupt_deltas(deltas, corruption_frac=0.5, seed=7)
        second = corrupt_deltas(deltas, corruption_frac=0.5, seed=7)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], deltas[0]))
        self.assertFalse(torch.equal(first, deltas))


class AggregationRuleTests(unittest.TestCase):
    def test_arithmetic_mean_uses_estimated_active_count(self):
        import torch

        counts = torch.tensor([[2.0, 1.0, 1.0]])
        weights = aggregate_sparse_vectors(counts, active_count=4)
        self.assertTrue(torch.equal(weights, torch.tensor([[0.5, 0.25, 0.25]])))

    def test_majority_voting_averages_tied_centroids(self):
        import torch

        counts = torch.tensor([[3.0, 3.0, 1.0]])
        weights = aggregate_sparse_vectors(counts, method="majority_voting", active_count=7)
        self.assertTrue(torch.equal(weights, torch.tensor([[0.5, 0.5, 0.0]])))

    def test_trimmed_mean_splits_cutoff_mass_across_ties(self):
        import torch

        counts = torch.tensor([[5.0, 3.0, 3.0, 3.0]])
        weights = aggregate_sparse_vectors(
            counts,
            method="trimmed_mean",
            trim_threshold=0.8,
            active_count=12,
        )
        expected = torch.tensor([[0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]])
        self.assertTrue(torch.allclose(weights, expected))


class IncompleteDatasetTests(unittest.TestCase):
    def test_incomplete_new_bundle_is_rejected_by_default(self):
        args = SimpleNamespace(allow_incomplete_dataset=False)
        bundle = {
            "meta": {"collection_complete": False, "fragment_remaining": {"train": 12}}
        }
        with patch.object(pretraining_module, "load_raw_ifed_bundle", return_value=("bundle.pth", bundle)):
            with self.assertRaisesRegex(ValueError, "collection is incomplete"):
                load_raw_splits(args, "cpu")


if __name__ == "__main__":
    unittest.main()
