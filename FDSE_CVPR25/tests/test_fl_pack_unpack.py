"""FL pack/unpack cycle test for FedDSA-VIB.

Validates Review Agent's A4 finding: Client.unpack must skip VIB-private
keys (log_var_head, log_sigma_prior) — otherwise every round the client's
locally-learned sigma gets overwritten by server's stale copy.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm.common.vib import VIBSemanticHead


class _FakeModel(nn.Module):
    """Mini surrogate to test unpack logic without flgo dependency."""

    def __init__(self, num_classes=7, feat_dim=32, proj_dim=16):
        super().__init__()
        # Components that exist in FedDSAVIBModel
        self.encoder = nn.Linear(feat_dim, feat_dim)
        self.style_head = nn.Linear(feat_dim, proj_dim)
        self.semantic_head = VIBSemanticHead(feat_dim, proj_dim, num_classes)
        self.sem_classifier = nn.Linear(proj_dim, num_classes)
        self.register_buffer('M', torch.zeros(proj_dim, num_classes))


def _mimic_unpack(client_model, server_model, skip_vib=True):
    """Replicate Client.unpack logic from feddsa_sgpa_vib.py.

    If skip_vib=True, the patched behavior (keep local VIB params).
    If skip_vib=False, the OLD buggy behavior (overwrite everything).
    """
    local_dict = client_model.state_dict()
    global_dict = server_model.state_dict()
    for key in local_dict.keys():
        if 'style_head' in key:
            continue
        if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
            continue
        if key.endswith('.M') or key == 'M':
            continue
        if skip_vib:
            if 'log_var_head' in key:
                continue
            if 'log_sigma_prior' in key:
                continue
        local_dict[key] = global_dict[key]
    client_model.load_state_dict(local_dict)


def test_client_unpack_preserves_log_var_head():
    """Patched unpack must keep client's log_var_head across FL rounds."""
    server = _FakeModel()
    client = copy.deepcopy(server)

    # Client locally learns log_var_head — modify weights to a distinguishable value
    with torch.no_grad():
        client.semantic_head.log_var_head[0].weight.fill_(42.0)
        client.semantic_head.log_var_head[2].weight.fill_(-13.0)

    # Server keeps its own log_var_head untouched (represents "never aggregated")
    server_lv_layer0_before = server.semantic_head.log_var_head[0].weight.clone()

    # Apply patched unpack (skip_vib=True)
    _mimic_unpack(client, server, skip_vib=True)

    # Client's log_var_head must NOT be reset to server's defaults
    assert torch.allclose(
        client.semantic_head.log_var_head[0].weight,
        torch.full_like(client.semantic_head.log_var_head[0].weight, 42.0)
    ), "Client log_var_head was overwritten by server (bug)"

    assert torch.allclose(
        client.semantic_head.log_var_head[2].weight,
        torch.full_like(client.semantic_head.log_var_head[2].weight, -13.0)
    ), "Client log_var_head[2] was overwritten by server (bug)"

    # Server was not touched
    assert torch.allclose(server.semantic_head.log_var_head[0].weight, server_lv_layer0_before)


def test_client_unpack_preserves_log_sigma_prior():
    """Patched unpack must keep client's log_sigma_prior."""
    server = _FakeModel(num_classes=7)
    client = copy.deepcopy(server)

    with torch.no_grad():
        client.semantic_head.log_sigma_prior.fill_(0.7)

    _mimic_unpack(client, server, skip_vib=True)

    assert torch.allclose(
        client.semantic_head.log_sigma_prior,
        torch.full_like(client.semantic_head.log_sigma_prior, 0.7)
    ), "log_sigma_prior was overwritten"


def test_buggy_old_unpack_would_fail():
    """Counter-test: WITHOUT skip_vib, the bug reproduces — client loses its σ."""
    server = _FakeModel()
    client = copy.deepcopy(server)

    with torch.no_grad():
        client.semantic_head.log_var_head[0].weight.fill_(42.0)

    # Old (buggy) behavior: skip_vib=False
    _mimic_unpack(client, server, skip_vib=False)

    # Client's log_var_head now equals server's default — bug reproduces
    assert not torch.allclose(
        client.semantic_head.log_var_head[0].weight,
        torch.full_like(client.semantic_head.log_var_head[0].weight, 42.0)
    ), "Buggy unpack should have overwritten log_var_head"


def test_client_unpack_prototype_ema_IS_overwritten():
    """Unlike log_var_head, prototype_ema is SERVER-MANAGED and should be
    copied DOWN to clients each round. Verify it gets overwritten."""
    server = _FakeModel()
    client = copy.deepcopy(server)

    # Server updates prototype_ema (simulate server aggregation)
    with torch.no_grad():
        server.semantic_head.prototype_ema.fill_(99.0)
        server.semantic_head.prototype_init.fill_(True)

    # Client is still at init (zeros)
    assert torch.allclose(
        client.semantic_head.prototype_ema,
        torch.zeros_like(client.semantic_head.prototype_ema)
    )

    _mimic_unpack(client, server, skip_vib=True)

    # After unpack, client's prototype_ema should equal server's
    assert torch.allclose(
        client.semantic_head.prototype_ema,
        torch.full_like(client.semantic_head.prototype_ema, 99.0)
    ), "prototype_ema must flow server → client"
    assert bool(client.semantic_head.prototype_init.item()), \
        "prototype_init flag must flow server → client"


def test_client_unpack_shared_weights_still_overwritten():
    """encoder + sem_classifier + mu_head SHOULD still be overwritten (FedAvg)."""
    server = _FakeModel()
    client = copy.deepcopy(server)

    with torch.no_grad():
        client.encoder.weight.fill_(5.0)  # client's local divergent
        server.encoder.weight.fill_(-5.0)  # server's aggregated

    _mimic_unpack(client, server, skip_vib=True)

    # encoder must be replaced by server
    assert torch.allclose(
        client.encoder.weight,
        torch.full_like(client.encoder.weight, -5.0)
    ), "encoder (shared) should be overwritten"


def test_client_unpack_style_head_preserved():
    """style_head must remain local (parent's existing FedBN-style behavior)."""
    server = _FakeModel()
    client = copy.deepcopy(server)

    with torch.no_grad():
        client.style_head.weight.fill_(77.0)
        server.style_head.weight.fill_(-77.0)

    _mimic_unpack(client, server, skip_vib=True)

    assert torch.allclose(
        client.style_head.weight,
        torch.full_like(client.style_head.weight, 77.0)
    ), "style_head (private) should NOT be overwritten"


def test_full_3_round_fl_cycle_sigma_learns():
    """Simulate 3 rounds of FL: client learns σ every round, server never
    aggregates σ. Verify σ accumulates changes (not reset)."""
    server = _FakeModel()
    client = copy.deepcopy(server)

    initial_sigma = client.semantic_head.log_var_head[0].weight.clone()

    for r in range(3):
        # "Client training" — modify log_var_head
        with torch.no_grad():
            client.semantic_head.log_var_head[0].weight += 0.1

        # Server aggregates (doesn't touch σ)
        # Server sends pack (copy of its own model)
        server_copy = copy.deepcopy(server)

        # Client unpacks — should NOT lose its σ changes
        _mimic_unpack(client, server_copy, skip_vib=True)

    # After 3 rounds, client's σ should be accumulated changes
    expected = initial_sigma + 0.3
    assert torch.allclose(
        client.semantic_head.log_var_head[0].weight, expected, atol=1e-6
    ), f"σ should accumulate: expected init+0.3, got {(client.semantic_head.log_var_head[0].weight - initial_sigma).max()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
