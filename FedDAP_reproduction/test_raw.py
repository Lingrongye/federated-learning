"""Original-file fixtures only; no network and no model training."""
import bz2
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from raw_data import (DOMAINS, RESOURCES, RawDigits, digest, download,
                      read_arrays, resource_paths, verify_sources)
from raw_loaders import loaders, transform
from run_digits import evaluate, parser
from wait_run import wait
from inspect_raw import inspect


class RawTests(unittest.TestCase):
    def test_source_preview_keeps_pixels_indices_and_refuses_overwrite(self):
        images = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        images[9, 0, 0, 0] = 255
        labels = np.arange(10)
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("inspect_raw.verify_sources", return_value={"splits": {}}), \
             mock.patch("inspect_raw.read_arrays", return_value=(images, labels)):
            inspect("/fixture", directory)
            path = Path(directory)
            with np.load(path / "source_samples.npz", allow_pickle=False) as item:
                self.assertEqual(len(item.files), 80)
                self.assertEqual(int(item["SYN_class9_index"]), 9)
                self.assertTrue(np.array_equal(item["SYN_class9_pixels"], images[9]))
            self.assertTrue((path / "source_preview.png").is_file())
            with self.assertRaises(FileExistsError):
                inspect("/fixture", directory)

    def test_completion_gate_failure_success_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                wait(root, timeout=.001, interval=.001)
            with open(root / "summary.json", "x") as stream:
                json.dump({"status": "completed"}, stream)
            self.assertEqual(wait(root)["status"], "completed")
            with open(root / "failure.json", "x") as stream:
                json.dump({"error": "controlled failure"}, stream)
            with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                wait(root)
            with self.assertRaises(ValueError):
                wait(root, timeout=0)

    def test_mnist_raw_idx_and_rgb_transform_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MNIST"
            path.mkdir()
            images = np.full((3, 28, 28), 255, dtype=np.uint8)
            labels = np.array([0, 5, 9], dtype=np.uint8)
            with gzip.open(path / "train-images-idx3-ubyte.gz", "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 3, 28, 28) + images.tobytes())
            with gzip.open(path / "train-labels-idx1-ubyte.gz", "wb") as stream:
                stream.write(struct.pack(">II", 2049, 3) + labels.tobytes())
            actual, target = read_arrays(directory, "MNIST", True)
            self.assertTrue(np.array_equal(images, actual))
            calls = mock.Mock(wraps=transform(False))
            dataset = RawDigits(actual, target, [2], calls)
            image, label = dataset[0]
            self.assertEqual(label, 9)
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(calls.call_count, 1)
            self.assertTrue(torch.allclose(image[:, 0, 0], torch.tensor([
                (1 - .485) / .229, (1 - .456) / .224, (1 - .406) / .225])))
            with self.assertRaises(ValueError):
                RawDigits(actual, target, [3], calls)

    def test_usps_matches_torchvision_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "USPS"
            path.mkdir()
            values = [-1., 0., 1.] + [-1.] * 253
            row = "10 " + " ".join(f"{i+1}:{value}" for i, value in enumerate(values)) + "\n"
            with bz2.open(path / "usps.bz2", "wt") as stream:
                stream.write(row)
            images, labels = read_arrays(directory, "USPS", True)
            self.assertEqual(labels.tolist(), [9])
            self.assertEqual(images[0].reshape(-1)[:3].tolist(), [0, 127, 255])

    def test_mat_shape_and_zero_label_mapping(self):
        for domain in ("SVHN", "SYN"):
            source = np.zeros((32, 32, 3, 2), dtype=np.uint8)
            source[0, 0, 0, 1] = 255
            with mock.patch("scipy.io.loadmat", return_value={"X": source, "y": np.array([[10], [8]])}):
                images, labels = read_arrays("/unused", domain, True)
            self.assertEqual(images.shape, (2, 32, 32, 3))
            self.assertEqual(labels.tolist(), [0, 8])
            self.assertEqual(int(images[1, 0, 0, 0]), 255)
        with mock.patch("scipy.io.loadmat", return_value={
            "X": source, "y": np.array([[0], [11]])}):
            with self.assertRaises(ValueError):
                read_arrays("/unused", "SYN", False)

    def test_four_domains_disjoint_indices_and_published_splits(self):
        args = parser().parse_args([
            "--data-format", "raw", "--data-root", "/original",
            "--dump-diag", "/unused", "--train-samples", "32", "--test-samples", "32",
        ])
        def arrays(root, domain, train):
            n = 120 if train else 50
            return np.zeros((n, 32, 32, 3), np.uint8), np.arange(n) % 10
        with mock.patch("raw_loaders.verify_sources", return_value={"status": "completed"}), \
             mock.patch("raw_loaders.digest", return_value="fixture"), \
             mock.patch("raw_loaders.read_arrays", side_effect=arrays):
            train, proto, test, manifest = loaders(args)
        self.assertEqual((len(train), len(proto), len(test)), (12, 12, 4))
        self.assertEqual(tuple(item["domain"] for item in manifest["test"]), DOMAINS)
        for domain in DOMAINS:
            sets = [set(item["indices"]) for item in manifest["clients"] if item["domain"] == domain]
            self.assertEqual(len(set.union(*sets)), 96)
            self.assertTrue(set(resource_paths(domain, True)).isdisjoint(resource_paths(domain, False)))
        self.assertEqual(sum(item["count"] for item in manifest["test"]), 128)

    def test_four_domain_evaluation_does_not_drop_syn(self):
        class Model:
            def eval(self):
                return self
            def features(self, x):
                return x
            def classifier(self, x):
                return x
        test = [[(torch.eye(10)[:2], torch.tensor([0, 1]))] for _ in DOMAINS]
        accuracy, snapshot = evaluate(Model(), test, "cpu", DOMAINS)
        self.assertEqual(accuracy.tolist(), [100.] * 4)
        self.assertEqual(snapshot["feature_domains"].tolist(), [d for d in DOMAINS for _ in range(2)])
        with self.assertRaises(ValueError):
            evaluate(Model(), test, "cpu", DOMAINS[:3])

    def test_incomplete_download_refused_and_failure_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new_download"
            with mock.patch("raw_data.urllib.request.urlopen", side_effect=OSError("network unavailable")):
                with self.assertRaises(RuntimeError):
                    download(root)
            self.assertTrue((root / "download_failure.json").exists())
            self.assertTrue((root / "download_events.jsonl").exists())
            with self.assertRaises(FileExistsError):
                download(root)
            with self.assertRaises(FileNotFoundError):
                verify_sources(root)

    def test_checksum_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "one.dat"
            file.write_bytes(b"verified fixture")
            fixture = [("one.dat", ["https://example.invalid"], "sha256", digest(file), 16)]
            with open(root / "download_manifest.json", "x") as stream:
                json.dump({"status": "completed", "files": [
                    {"relative": "one.dat", "sha256": digest(file), "bytes": 16}]}, stream)
            with mock.patch("raw_data.RESOURCES", fixture):
                verify_sources(root)
                file.write_bytes(b"tampered fixture")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_sources(root)
        self.assertEqual(len(RESOURCES), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
