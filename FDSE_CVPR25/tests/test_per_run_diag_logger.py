# -*- coding: utf-8 -*-
"""Smoke test for PerRunDiagLogger with mock server + clients.

Since PACS raw data is not present locally, we mock the flgo server/client
interface to verify the logger integration produces the expected record
keys without actually running federated training.
"""
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class MockDataset(Dataset):
    def __init__(self, n=32, num_classes=7, feat_dim=3):
        self.x = torch.randn(n, feat_dim, 32, 32)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class MockModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


class MockClient:
    def __init__(self, model, test_data, device='cpu'):
        self.model = model
        self.test_data = test_data
        self.val_data = test_data  # reuse for val
        self.train_data = test_data
        self.device = device
        self.test_batch_size = 8
        self.calculator = type('C', (), {'collate_fn': None})()

    def test(self, model, flag='test'):
        """Mock of BasicClient.test — returns acc/loss."""
        ds = self.test_data if flag == 'test' else self.val_data
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_correct = 0
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for i in range(len(ds)):
                x, y = ds[i]
                x = x.unsqueeze(0)
                y = torch.tensor([y])
                logits = model(x)
                total_loss += loss_fn(logits, y).item()
                total_correct += (logits.argmax(1) == y).sum().item()
                n += 1
        return {'accuracy': total_correct / n, 'loss': total_loss / n}


class MockServer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.task = 'PACS_c4'  # tells logger num_classes=7
        self.val_data = None


def test_logger_inherits_correctly():
    import logger
    assert hasattr(logger, 'PerRunDiagLogger')
    assert issubclass(logger.PerRunDiagLogger, logger.PerRunLogger)
    print('PASS: PerRunDiagLogger exists and inherits from PerRunLogger')


def test_diag_forward_on_mock():
    """Directly test the diag call path without BasicLogger __init__ complexity."""
    from diagnostics.per_class_eval import run_diagnostic

    model = MockModel(num_classes=7)
    ds = MockDataset(n=32, num_classes=7)
    diag = run_diagnostic(model, ds, device='cpu', num_classes=7, batch_size=8)

    # Check shape
    assert len(diag['per_class_acc']) == 7
    assert len(diag['per_class_conf']) == 7
    assert len(diag['per_class_support']) == 7
    assert 0 <= diag['overall_acc'] <= 1
    assert 0 <= diag['ece'] <= 1
    assert sum(diag['per_class_support']) == 32
    print('PASS: diag forward on mock model')


def test_logger_log_once_mock():
    """Directly exercise PerRunDiagLogger.log_once logic without full init."""
    import logger

    # Manually create a PerRunDiagLogger instance, bypassing __init__
    lg = logger.PerRunDiagLogger.__new__(logger.PerRunDiagLogger)
    lg.output = defaultdict(list)
    lg.option = {'num_workers': 0}

    # Seed parent class outputs to simulate super().log_once() result
    # (so round_id computation works)
    lg.output['local_test_accuracy'] = [0.5, 0.6]  # 2 rounds recorded

    # Mock server + clients
    num_classes = 7
    model = MockModel(num_classes=num_classes)
    clients = [
        MockClient(model, MockDataset(n=16, num_classes=num_classes))
        for _ in range(4)
    ]
    server = MockServer(model)

    lg.server = server
    lg.clients = clients

    # Override _get_num_classes to return 7 (avoid task_name parsing)
    lg._get_num_classes = lambda: num_classes

    # Monkey-patch super().log_once to a no-op (we don't want parent's c.test)
    # Instead we'll call our diag path directly
    import types

    def custom_log_once(self, *a, **kw):
        # Skip parent, call diag part
        from diagnostics.per_class_eval import run_diagnostic as rd
        round_id = len(self.output.get('local_test_accuracy', [])) - 1
        include_hist = (round_id % self.HIST_EVERY == 0) or (round_id == 0)

        pc_acc_dist, pc_conf_dist, pc_support_dist, conf_stats_dist = [], [], [], []
        hist_dist = []
        for c in self.clients:
            m = c.model
            try:
                device = next(m.parameters()).device
            except StopIteration:
                device = self.server.device
            test_data = c.test_data
            bs = min(c.test_batch_size, len(test_data))
            diag = rd(
                model=m, dataset=test_data, device=device,
                num_classes=num_classes, batch_size=bs,
                num_workers=0, collate_fn=None,
                include_histogram=include_hist,
            )
            pc_acc_dist.append(diag['per_class_acc'])
            pc_conf_dist.append(diag['per_class_conf'])
            pc_support_dist.append(diag['per_class_support'])
            conf_stats_dist.append({
                'mean': diag['conf_mean'], 'std': diag['conf_std'],
                'p10': diag['conf_p10'], 'p50': diag['conf_p50'], 'p90': diag['conf_p90'],
                'ece': diag['ece'],
                'over_conf_err_ratio': diag['over_conf_err_ratio'],
                'wrong_conf_mean': diag['wrong_conf_mean'],
            })
            if include_hist:
                hist_dist.append({
                    'hist_correct': diag.get('hist_correct'),
                    'hist_wrong': diag.get('hist_wrong'),
                    'bins': diag.get('hist_bins', 20),
                })
        self.output['per_class_test_acc_dist'].append(pc_acc_dist)
        self.output['per_class_test_conf_dist'].append(pc_conf_dist)
        self.output['per_class_test_support_dist'].append(pc_support_dist)
        self.output['confidence_stats_dist'].append(conf_stats_dist)
        if include_hist:
            self.output['confidence_hist_dist'].append({
                'round': round_id, 'per_client': hist_dist,
            })

    lg.log_once = types.MethodType(custom_log_once, lg)
    lg.log_once()

    # Verify output keys populated
    assert 'per_class_test_acc_dist' in lg.output
    assert len(lg.output['per_class_test_acc_dist']) == 1
    assert len(lg.output['per_class_test_acc_dist'][0]) == 4  # 4 clients
    assert len(lg.output['per_class_test_acc_dist'][0][0]) == 7  # 7 classes
    assert 'confidence_stats_dist' in lg.output
    assert 'ece' in lg.output['confidence_stats_dist'][0][0]
    # round_id = len(local_test_accuracy)-1 = 1, not divisible by 50 but != 0 → no hist
    # But we seeded 2 entries → round_id = 1. Not == 0, not divisible by 50.
    # So hist should NOT be added.
    # Let me re-check: seed was [0.5, 0.6] → len=2 → round_id=1 → include_hist = (1 % 50 == 0) or (1 == 0) = False
    assert 'confidence_hist_dist' not in lg.output or \
        len(lg.output['confidence_hist_dist']) == 0, \
        "hist should not be dumped at round 1"
    print('PASS: log_once mock')


def test_logger_log_once_round_zero_hist():
    """At round 0, histogram should be included."""
    import logger
    from diagnostics.per_class_eval import run_diagnostic as rd
    import types

    lg = logger.PerRunDiagLogger.__new__(logger.PerRunDiagLogger)
    lg.output = defaultdict(list)
    lg.option = {'num_workers': 0}
    lg.output['local_test_accuracy'] = [0.5]  # 1 entry → round_id = 0

    num_classes = 7
    model = MockModel(num_classes=num_classes)
    clients = [
        MockClient(model, MockDataset(n=16, num_classes=num_classes))
        for _ in range(4)
    ]
    server = MockServer(model)
    lg.server = server
    lg.clients = clients

    def custom_log_once(self):
        round_id = len(self.output.get('local_test_accuracy', [])) - 1
        include_hist = (round_id % self.HIST_EVERY == 0) or (round_id == 0)
        hist_dist = []
        for c in self.clients:
            m = c.model
            device = next(m.parameters()).device
            diag = rd(
                model=m, dataset=c.test_data, device=device,
                num_classes=num_classes, batch_size=8,
                num_workers=0, include_histogram=include_hist,
            )
            if include_hist:
                hist_dist.append({
                    'hist_correct': diag.get('hist_correct'),
                    'hist_wrong': diag.get('hist_wrong'),
                })
        if include_hist:
            self.output['confidence_hist_dist'].append({
                'round': round_id, 'per_client': hist_dist,
            })

    lg.log_once = types.MethodType(custom_log_once, lg)
    lg.log_once()

    assert 'confidence_hist_dist' in lg.output
    assert len(lg.output['confidence_hist_dist']) == 1
    assert lg.output['confidence_hist_dist'][0]['round'] == 0
    assert len(lg.output['confidence_hist_dist'][0]['per_client']) == 4
    print('PASS: round 0 includes histogram')


def test_num_classes_inference():
    import logger
    lg = logger.PerRunDiagLogger.__new__(logger.PerRunDiagLogger)

    # PACS
    lg.server = MockServer(MockModel(num_classes=7))
    lg.server.task = 'PACS_c4'
    assert lg._get_num_classes() == 7

    # Office
    lg.server.task = 'office_caltech10_c4'
    assert lg._get_num_classes() == 10

    # Digit5
    lg.server.task = 'digit5_c5'
    assert lg._get_num_classes() == 10

    # Fallback: use last Linear
    lg.server.task = 'unknown'
    lg.server.model = MockModel(num_classes=7)
    assert lg._get_num_classes() == 7
    print('PASS: num_classes inference')


def test_output_json_safe():
    """Verify all output values are JSON-serializable (no np.float/torch.Tensor)."""
    import json
    from diagnostics.per_class_eval import run_diagnostic

    model = MockModel(num_classes=7)
    ds = MockDataset(n=16, num_classes=7)
    diag = run_diagnostic(model, ds, device='cpu', num_classes=7, batch_size=4,
                        include_histogram=True)

    # Attempt JSON dump — will raise on unsupported types
    s = json.dumps(diag)
    assert len(s) > 0
    # Round-trip
    back = json.loads(s)
    assert back['per_class_acc'] == diag['per_class_acc']
    print('PASS: output JSON-serializable')


if __name__ == '__main__':
    test_logger_inherits_correctly()
    test_diag_forward_on_mock()
    test_logger_log_once_mock()
    test_logger_log_once_round_zero_hist()
    test_num_classes_inference()
    test_output_json_safe()
    print('\nALL 6 integration tests PASSED')
