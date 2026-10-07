"""Run locally BEFORE training: python FedDAP_reproduction/test_core.py."""
import ast
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def syntax_check():
    for source in HERE.glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    print("AST_PASS", flush=True)


syntax_check()
if "--syntax-only" in sys.argv:
    sys.exit(0)

import math
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from core import (aggregate_states, alignment_losses, attention_aggregate,
                  extract_prototypes)
from run_digits import CachedDigits, backbone, parser, run, save_npz

torch.set_num_threads(1)


class CoreTests(unittest.TestCase):
    def test_cpcl_formula_and_detached_gradient(self):
        feature = torch.tensor([[0.6, 0.8]], requires_grad=True)
        positive = torch.tensor([1., 0.], requires_grad=True)
        negative = torch.tensor([0., 1.], requires_grad=True)
        own = torch.tensor([-1., 0.], requires_grad=True)
        refs = {(0, "other"): positive, (1, "other"): negative,
                (0, "own"): own, (1, "own"): own}
        dpa, cpcl, nd, nc = alignment_losses(feature, torch.tensor([0]), "own", refs, 0.2)
        expected = -math.log(math.exp(0.6 / 0.2) /
                            (math.exp(0.6 / 0.2) + math.exp(0.8 / 0.2)))
        self.assertAlmostEqual(float(cpcl.detach()), expected, places=5)
        self.assertAlmostEqual(float(dpa.detach()), 1.6, places=5)
        self.assertEqual((nd, nc), (1, 1))
        self.assertGreater(float(torch.autograd.grad(dpa, feature, retain_graph=True)[0].norm()), 0)
        self.assertGreater(float(torch.autograd.grad(cpcl, feature, retain_graph=True)[0].norm()), 0)
        (dpa + cpcl).backward()
        self.assertGreater(float(feature.grad.norm()), 0)
        self.assertTrue(all(value.grad is None for value in refs.values()))

    def test_cpcl_multiple_positives(self):
        feature = torch.tensor([[1., 0.]], requires_grad=True)
        refs = {(0, "b"): torch.tensor([1., 0.]),
                (0, "c"): torch.tensor([0., 1.]),
                (1, "c"): torch.tensor([-1., 0.])}
        _, loss, _, _ = alignment_losses(feature, torch.tensor([0]), "a", refs, 1.)
        expected = math.log(math.e + 1 + 1 / math.e) - math.log(math.e + 1)
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)

    def test_missing_anchors_zero_with_gradient(self):
        feature = torch.randn(3, 4, requires_grad=True)
        dpa, cpcl, nd, nc = alignment_losses(feature, torch.tensor([0, 1, 2]), "a", {}, .02)
        (dpa + cpcl).backward()
        self.assertEqual((float(dpa), float(cpcl), nd, nc), (0., 0., 0, 0))
        self.assertTrue(torch.equal(feature.grad, torch.zeros_like(feature)))

    def test_extreme_temperature_is_finite(self):
        features = torch.tensor([[1., 0.]], requires_grad=True)
        refs = {(0, "b"): torch.tensor([-1., 0.]), (1, "b"): torch.tensor([1., 0.])}
        dpa, cpcl, _, _ = alignment_losses(features, torch.tensor([0]), "a", refs, 1e-5)
        self.assertTrue(torch.isfinite(cpcl))
        (dpa + cpcl).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        with self.assertRaises(ValueError):
            alignment_losses(features, torch.tensor([0]), "a", refs, 0)

    def test_attention_known_weights_and_domain_isolation(self):
        items = [{(0, "a"): torch.tensor([1., 0.])},
                 {(0, "a"): torch.tensor([1., 0.])},
                 {(0, "a"): torch.tensor([0., 1.]), (0, "b"): torch.tensor([4., 3.])}]
        result, diag = attention_aggregate(items, 1.)
        expected = torch.softmax(torch.tensor([1., 1., 0.]), 0)
        self.assertTrue(torch.allclose(torch.tensor(diag[(0, "a")]["weights"]), expected))
        self.assertTrue(torch.allclose(result[(0, "a")], torch.tensor([expected[0] * 2, expected[2]])))
        self.assertTrue(torch.equal(result[(0, "b")], torch.tensor([4., 3.])))
        self.assertEqual(diag[(0, "b")]["weights"], [1.])
        with self.assertRaises(ValueError):
            attention_aggregate(items, 0)

    def test_fedavg_sample_weighting_and_bn_counter(self):
        result, weights = aggregate_states([
            {"weight": torch.tensor([1.]), "bn.running_mean": torch.tensor([2.]),
             "bn.num_batches_tracked": torch.tensor(3)},
            {"weight": torch.tensor([5.]), "bn.running_mean": torch.tensor([6.]),
             "bn.num_batches_tracked": torch.tensor(7)},
        ], [1, 3])
        self.assertEqual(weights, [.25, .75])
        self.assertEqual(float(result["weight"]), 4)
        self.assertEqual(float(result["bn.running_mean"]), 5)
        self.assertEqual(int(result["bn.num_batches_tracked"]), 7)
        with self.assertRaises(ValueError):
            aggregate_states([], [])

    def test_final_encoder_prototypes_full_pass_and_mode_restore(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
            def features(self, x):
                return x * self.scale
        model = Encoder().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        optimizer.zero_grad()
        (model.features(torch.ones(2, 2)).sum()).backward()
        optimizer.step()
        inputs = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 0, 1])), batch_size=2)
        result, counts = extract_prototypes(model, loader, "a", "cpu")
        self.assertTrue(model.training)
        self.assertEqual(sum(counts.values()), 3)
        self.assertTrue(torch.allclose(result[(0, "a")], inputs[:2].mean(0) * model.scale.detach()))
        self.assertTrue(all(not value.requires_grad for value in result.values()))

    def test_resnet_forward_features_classifier_gradient(self):
        net, _ = backbone()
        images = torch.randn(2, 3, 32, 32)
        net.eval()
        self.assertTrue(torch.allclose(net(images), net.classifier(net.features(images))))
        net.train()
        loss = F.cross_entropy(net.classifier(net.features(images)), torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(float(net.conv1.weight.grad.norm()), 0)
        self.assertGreater(float(net.cls.weight.grad.norm()), 0)

    def test_backbone_upstream_executable_ast(self):
        upstream = ROOT / "FedDAP_CVPR2026/backbone/ResNet.py"
        if not upstream.exists():
            self.skipTest("Upstream clone exists locally only; verified before Git deployment")
        candidate = ROOT / "F2DC/backbone/ResNet.py"
        self.assertEqual(ast.dump(ast.parse(upstream.read_text())),
                         ast.dump(ast.parse(candidate.read_text())))

    def test_real_cache_adapter_shape_and_single_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trusted_fixture.pkl"
            container = np.empty(2, dtype=object)
            container[0] = np.zeros((3, 28, 28), dtype=np.uint8)
            container[1] = np.array([0, 1, 2])
            with open(source, "xb") as stream:
                np.save(stream, container, allow_pickle=True)
            dataset = CachedDigits(source, [2, 0])
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(label, 2)
            self.assertTrue(torch.isfinite(image).all())

    def test_diagnostics_exclusive_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_001.npz"
            save_npz(path, witness=np.array([1]))
            with self.assertRaises(FileExistsError):
                save_npz(path, witness=np.array([2]))
            self.assertEqual(int(np.load(path)["witness"][0]), 1)

    def test_environment_failure_is_preserved_and_cannot_rerun_over_it(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory) / "diag_failure_witness"
            args = parser().parse_args([
                "--device", "cpu", "--data-root", directory, "--dump-diag", str(diag)
            ])
            with mock.patch("run_digits.backbone", side_effect=RuntimeError("controlled failure")):
                with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                    run(args)
            self.assertTrue((diag / "attempt.json").exists())
            self.assertTrue((diag / "failure.json").exists())
            original = (diag / "failure.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual((diag / "failure.json").read_bytes(), original)

    def test_cuda_unavailable_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory) / "diag_cuda_failure_witness"
            args = parser().parse_args([
                "--device", "cuda:0", "--data-root", directory, "--dump-diag", str(diag)
            ])
            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
                    run(args)
            self.assertTrue((diag / "attempt.json").exists())
            self.assertTrue((diag / "failure.json").exists())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
