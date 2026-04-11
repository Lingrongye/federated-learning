"""
Unit tests for feddsa_regime_gated.

Tests the pure logic of:
    1. _compute_regime_score
    2. _apply_sam_lookahead
    3. _aggregate_shared_consensus flow (pseudo-gradient tracking)

Bypasses the flgo framework init by instantiating via __new__
and injecting required attributes manually.

Run from FDSE_CVPR25 directory:
    python ../experiments/ablation/EXP-067_regime_gated/test_regime_gated.py
"""
import sys
import os
import copy
import math

# Ensure we can import from FDSE_CVPR25/algorithm
HERE = os.path.dirname(os.path.abspath(__file__))
FDSE_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'FDSE_CVPR25'))
sys.path.insert(0, FDSE_ROOT)

import torch
import numpy as np

# Import only what we need (avoid flgo framework init)
from algorithm.feddsa_regime_gated import Server as RGServer


def make_mock_server(regime_threshold=0.0, sam_rho=0.05):
    """Create a Server instance bypassing flgo init."""
    s = RGServer.__new__(RGServer)
    s.style_bank = {}
    s.regime_threshold = regime_threshold
    s.sam_rho = sam_rho
    s.prev_pseudo_grad = None
    s.regime_history = []
    s.current_round = 0
    # Dummy shared_keys used only in _aggregate_shared_consensus test
    s.shared_keys = []
    return s


def test_regime_score_empty_bank():
    s = make_mock_server()
    assert s._compute_regime_score() is None, "empty bank should return None"
    print("  PASS  regime_score_empty_bank")


def test_regime_score_single_client():
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    assert s._compute_regime_score() is None, "single client should return None"
    print("  PASS  regime_score_single_client")


def test_regime_score_two_identical():
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s.style_bank[1] = (torch.zeros(4), torch.ones(4))
    r = s._compute_regime_score()
    assert r is not None and abs(r) < 1e-5, f"identical clients should give r≈0, got {r}"
    print(f"  PASS  regime_score_two_identical (r={r})")


def test_regime_score_two_different():
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    # mu offset 2.0, sigma 2x → log_sigma diff log(2.0)
    s.style_bank[1] = (torch.ones(4) * 2.0, torch.ones(4) * 2.0)
    r = s._compute_regime_score()
    # mu_d = ||[2,2,2,2]|| = sqrt(16) = 4.0
    # log sigma diff = log(2) - log(1) = 0.693 per component
    # sig_d = ||[0.693]*4|| = sqrt(4*0.693^2) = 2*0.693 = 1.386
    # total = 4.0 + 1.386 = 5.386
    expected = 4.0 + 2 * math.log(2.0)
    assert r is not None and abs(r - expected) < 1e-4, f"expected ~{expected}, got {r}"
    print(f"  PASS  regime_score_two_different (r={r:.4f}, expected {expected:.4f})")


def test_regime_score_mean_of_pairs():
    """With 3 clients, regime is mean of 3 pairs."""
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(2), torch.ones(2))
    s.style_bank[1] = (torch.ones(2), torch.ones(2))    # mu_d=sqrt(2), sig=0
    s.style_bank[2] = (torch.ones(2) * 2.0, torch.ones(2))  # vs 0: mu_d=sqrt(8)
    # pairs: (0,1): sqrt(2), (0,2): sqrt(8), (1,2): sqrt(2)
    expected = (math.sqrt(2) + math.sqrt(8) + math.sqrt(2)) / 3
    r = s._compute_regime_score()
    assert abs(r - expected) < 1e-4, f"expected {expected}, got {r}"
    print(f"  PASS  regime_score_mean_of_pairs (r={r:.4f})")


def test_regime_score_pacs_vs_office_ordering():
    """Synthetic: PACS-like (large gaps) should give larger r than Office-like (small gaps)."""
    s_pacs = make_mock_server()
    # 4 clients with large pairwise distance
    for i in range(4):
        mu = torch.ones(8) * (i * 5.0)  # spacing 5.0
        sigma = torch.ones(8) * (1.0 + i * 0.3)
        s_pacs.style_bank[i] = (mu, sigma)
    r_pacs = s_pacs._compute_regime_score()

    s_off = make_mock_server()
    for i in range(4):
        mu = torch.ones(8) * (i * 0.1)  # tiny spacing
        sigma = torch.ones(8) * (1.0 + i * 0.01)
        s_off.style_bank[i] = (mu, sigma)
    r_office = s_off._compute_regime_score()

    assert r_pacs > r_office * 5, \
        f"PACS-like r ({r_pacs}) should be >> Office-like r ({r_office})"
    print(f"  PASS  pacs_vs_office_ordering (r_pacs={r_pacs:.3f}, r_office={r_office:.3f})")


def test_sam_lookahead_no_prev():
    """If no prev_pseudo_grad, lookahead should be a no-op."""
    s = make_mock_server()
    # Mock model via state_dict - simulate with simple tensor storage

    class MockModel:
        def __init__(self, state):
            self._state = state
        def state_dict(self):
            return {k: v.clone() for k, v in self._state.items()}
        def load_state_dict(self, d, strict=False):
            for k, v in d.items():
                self._state[k] = v.clone()

    s.model = MockModel({'w': torch.tensor([1.0, 2.0, 3.0])})
    s.prev_pseudo_grad = None
    before = s.model.state_dict()['w'].clone()
    s._apply_sam_lookahead()
    after = s.model.state_dict()['w']
    assert torch.allclose(before, after), "no prev_pseudo_grad should be a no-op"
    print("  PASS  sam_lookahead_no_prev")


def test_sam_lookahead_applies_perturbation():
    """With prev_pseudo_grad set, model should move by rho * normalized prev."""
    s = make_mock_server(sam_rho=0.1)

    class MockModel:
        def __init__(self, state):
            self._state = state
        def state_dict(self):
            return {k: v.clone() for k, v in self._state.items()}
        def load_state_dict(self, d, strict=False):
            for k, v in d.items():
                self._state[k] = v.clone()

    w0 = torch.tensor([1.0, 2.0, 3.0])
    s.model = MockModel({'w': w0.clone()})
    g = torch.tensor([2.0, 0.0, 0.0])  # norm = 2.0
    s.prev_pseudo_grad = {'w': g.clone()}
    s._apply_sam_lookahead()

    after = s.model.state_dict()['w']
    # Expected: w0 + (0.1 / 2.0) * g = [1 + 0.1, 2, 3] = [1.1, 2, 3]
    expected = torch.tensor([1.1, 2.0, 3.0])
    assert torch.allclose(after, expected, atol=1e-5), \
        f"expected {expected}, got {after}"
    print(f"  PASS  sam_lookahead_applies_perturbation (after={after.tolist()})")


def test_sam_lookahead_direction_matches_prev():
    """SAM step must be ALONG the prev direction (positive sign)."""
    s = make_mock_server(sam_rho=0.5)

    class MockModel:
        def __init__(self, state):
            self._state = state
        def state_dict(self):
            return {k: v.clone() for k, v in self._state.items()}
        def load_state_dict(self, d, strict=False):
            for k, v in d.items():
                self._state[k] = v.clone()

    s.model = MockModel({
        'a': torch.zeros(3),
        'b': torch.zeros(2),
    })
    s.prev_pseudo_grad = {
        'a': torch.tensor([3.0, 4.0, 0.0]),  # norm 5
        'b': torch.tensor([0.0, 0.0]),
    }
    # Total norm = sqrt(25 + 0) = 5
    s._apply_sam_lookahead()
    after = s.model.state_dict()
    # a: 0 + (0.5/5) * [3,4,0] = [0.3, 0.4, 0]
    expected_a = torch.tensor([0.3, 0.4, 0.0])
    assert torch.allclose(after['a'], expected_a, atol=1e-5), \
        f"expected a={expected_a}, got {after['a']}"
    # b: should be zero
    assert torch.allclose(after['b'], torch.zeros(2)), f"b should stay 0, got {after['b']}"
    print(f"  PASS  sam_lookahead_direction_matches_prev")


def test_regime_gate_logic():
    """Verify the gate decision: low r + has prev → strategy='consensus+sam',
       otherwise strategy='consensus'.

       We test the gate in isolation by directly inspecting the regime_history
       after manually calling _aggregate_shared_consensus with mocks. But since
       that method requires full model state, we instead simulate the gate.
    """
    s = make_mock_server(regime_threshold=1.0)
    # Case 1: r < threshold + has prev → should trigger SAM
    s.prev_pseudo_grad = {'w': torch.tensor([1.0, 0.0])}
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s.style_bank[1] = (torch.ones(4) * 0.1, torch.ones(4) * 1.01)
    r = s._compute_regime_score()
    assert r < 1.0, f"r={r} should be < 1.0 for this test"
    # Gate logic check
    should_trigger = (r is not None and r < s.regime_threshold
                      and s.prev_pseudo_grad is not None)
    assert should_trigger, "low r + prev should trigger SAM"

    # Case 2: r > threshold → no trigger
    s2 = make_mock_server(regime_threshold=1.0)
    s2.prev_pseudo_grad = {'w': torch.tensor([1.0, 0.0])}
    s2.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s2.style_bank[1] = (torch.ones(4) * 10.0, torch.ones(4) * 5.0)
    r2 = s2._compute_regime_score()
    assert r2 > 1.0, f"r2={r2} should be >> 1.0 for this test"
    should_trigger2 = (r2 is not None and r2 < s2.regime_threshold
                       and s2.prev_pseudo_grad is not None)
    assert not should_trigger2, "high r should NOT trigger SAM"

    # Case 3: prev None → no trigger
    s3 = make_mock_server(regime_threshold=1.0)
    s3.prev_pseudo_grad = None
    s3.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s3.style_bank[1] = (torch.ones(4) * 0.1, torch.ones(4) * 1.01)
    r3 = s3._compute_regime_score()
    should_trigger3 = (r3 is not None and r3 < s3.regime_threshold
                       and s3.prev_pseudo_grad is not None)
    assert not should_trigger3, "no prev should NOT trigger SAM (first round)"

    print(f"  PASS  regime_gate_logic (r_low={r:.3f}, r_high={r2:.3f})")


def test_method_inheritance():
    """Verify Server inherits consensus aggregation from feddsa_consensus."""
    s = make_mock_server()
    # Must have parent's consensus method available (don't call it, just check)
    assert hasattr(s, '_aggregate_shared_consensus'), \
        "must inherit _aggregate_shared_consensus"
    assert hasattr(s, '_optim_lambda'), "must inherit _optim_lambda from consensus parent"
    assert hasattr(s, '_quadprog'), "must inherit _quadprog from consensus parent"
    # New graph dispatch methods
    assert hasattr(s, '_dispatch_knn_styles'), "must have _dispatch_knn_styles"
    assert hasattr(s, '_style_graph_edges'), "must have _style_graph_edges"
    print("  PASS  method_inheritance")


def test_knn_dispatch_picks_nearest():
    """KNN dispatch should pick the k nearest styles to the query client."""
    s = make_mock_server()
    # Client 0 at origin, client 1 at dist 1, client 2 at dist 10, client 3 at dist 5
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s.style_bank[1] = (torch.ones(4) * 0.5, torch.ones(4))   # dist ~1.0
    s.style_bank[2] = (torch.ones(4) * 5.0, torch.ones(4))   # dist ~10.0
    s.style_bank[3] = (torch.ones(4) * 2.5, torch.ones(4))   # dist ~5.0

    available = {cid: s.style_bank[cid] for cid in [1, 2, 3]}
    chosen = s._dispatch_knn_styles(client_id=0, available=available, k=2)

    # Should return 2 items
    assert len(chosen) == 2, f"expected 2, got {len(chosen)}"
    # Should match the nearest two: client 1 (dist ~1) and client 3 (dist ~5)
    # Check by matching mu values
    chosen_mus = [c[0][0].item() for c in chosen]  # first dim of mu
    assert 0.5 in chosen_mus, f"should include client 1's mu 0.5, got {chosen_mus}"
    assert 2.5 in chosen_mus, f"should include client 3's mu 2.5, got {chosen_mus}"
    assert 5.0 not in chosen_mus, f"should NOT include client 2's mu 5.0, got {chosen_mus}"
    print(f"  PASS  knn_dispatch_picks_nearest (mus={chosen_mus})")


def test_knn_dispatch_excludes_self_via_available():
    """Self exclusion is handled by the caller (pack builds `available` without self).
    Verify that when `available` already excludes the client, the function doesn't
    accidentally include it by looking it up via client_id."""
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s.style_bank[1] = (torch.ones(4), torch.ones(4))
    s.style_bank[2] = (torch.ones(4) * 2.0, torch.ones(4))

    # available does NOT include client 0
    available = {1: s.style_bank[1], 2: s.style_bank[2]}
    chosen = s._dispatch_knn_styles(client_id=0, available=available, k=2)
    chosen_mus = sorted([c[0][0].item() for c in chosen])
    assert chosen_mus == [1.0, 2.0], f"expected [1, 2], got {chosen_mus}"
    # Verify 0 is nowhere
    assert 0.0 not in chosen_mus
    print(f"  PASS  knn_dispatch_excludes_self_via_available")


def test_knn_dispatch_fallback_no_self_bank():
    """When the querying client has no style in the bank yet (fresh client),
    fall back to random — still returns k items, no crash."""
    s = make_mock_server()
    # Only clients 1, 2, 3 have styles
    s.style_bank[1] = (torch.ones(4), torch.ones(4))
    s.style_bank[2] = (torch.ones(4) * 2.0, torch.ones(4))
    s.style_bank[3] = (torch.ones(4) * 3.0, torch.ones(4))

    available = {1: s.style_bank[1], 2: s.style_bank[2], 3: s.style_bank[3]}
    # Client 0 has no entry → should fall back to random
    np.random.seed(42)
    chosen = s._dispatch_knn_styles(client_id=0, available=available, k=2)
    assert len(chosen) == 2, f"expected 2, got {len(chosen)}"
    print(f"  PASS  knn_dispatch_fallback_no_self_bank")


def test_knn_dispatch_k_larger_than_bank():
    """If k > available count, should return all available (degraded gracefully)."""
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s.style_bank[1] = (torch.ones(4), torch.ones(4))

    available = {1: s.style_bank[1]}
    chosen = s._dispatch_knn_styles(client_id=0, available=available, k=5)
    assert len(chosen) == 1, f"expected 1 (only 1 available), got {len(chosen)}"
    print(f"  PASS  knn_dispatch_k_larger_than_bank")


def test_style_graph_edges_count():
    """N clients → N*(N-1)/2 edges."""
    s = make_mock_server()
    for i in range(5):
        s.style_bank[i] = (torch.ones(3) * i, torch.ones(3))
    edges = s._style_graph_edges()
    expected = 5 * 4 // 2
    assert len(edges) == expected, f"expected {expected}, got {len(edges)}"
    # Check that distance ordering is correct: (0,1)=sqrt(3)=1.732, (0,4)=sqrt(48)=6.93
    # Find edge (0,1) and (0,4)
    d01 = next(d for i, j, d in edges if i == 0 and j == 1)
    d04 = next(d for i, j, d in edges if i == 0 and j == 4)
    assert d04 > d01, f"d04={d04} should > d01={d01}"
    print(f"  PASS  style_graph_edges_count (N=5, |E|={len(edges)}, d01={d01:.3f}, d04={d04:.3f})")


def test_knn_dispatch_agrees_with_graph_edges():
    """The k-nearest returned by _dispatch_knn_styles should be the smallest-k
    edges from the query client in the graph."""
    s = make_mock_server()
    s.style_bank[0] = (torch.zeros(4), torch.ones(4))
    s.style_bank[1] = (torch.ones(4) * 10, torch.ones(4))  # far
    s.style_bank[2] = (torch.ones(4) * 2, torch.ones(4))   # mid
    s.style_bank[3] = (torch.ones(4) * 1, torch.ones(4))   # near
    s.style_bank[4] = (torch.ones(4) * 5, torch.ones(4))   # mid-far

    available = {c: s.style_bank[c] for c in [1, 2, 3, 4]}
    chosen = s._dispatch_knn_styles(client_id=0, available=available, k=2)

    # From client 0, nearest 2 should be clients 3 (dist ~2) and 2 (dist ~4)
    chosen_mus = sorted([c[0][0].item() for c in chosen])
    assert chosen_mus == [1.0, 2.0], f"expected [1.0, 2.0], got {chosen_mus}"
    print(f"  PASS  knn_dispatch_agrees_with_graph_edges")


def run_all():
    tests = [
        test_regime_score_empty_bank,
        test_regime_score_single_client,
        test_regime_score_two_identical,
        test_regime_score_two_different,
        test_regime_score_mean_of_pairs,
        test_regime_score_pacs_vs_office_ordering,
        test_sam_lookahead_no_prev,
        test_sam_lookahead_applies_perturbation,
        test_sam_lookahead_direction_matches_prev,
        test_regime_gate_logic,
        test_method_inheritance,
        test_knn_dispatch_picks_nearest,
        test_knn_dispatch_excludes_self_via_available,
        test_knn_dispatch_fallback_no_self_bank,
        test_knn_dispatch_k_larger_than_bank,
        test_style_graph_edges_count,
        test_knn_dispatch_agrees_with_graph_edges,
    ]
    n_pass, n_fail = 0, 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            n_fail += 1

    total = n_pass + n_fail
    print(f"\n  Summary: {n_pass}/{total} passed")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    print("Running feddsa_regime_gated unit tests...")
    sys.exit(run_all())
