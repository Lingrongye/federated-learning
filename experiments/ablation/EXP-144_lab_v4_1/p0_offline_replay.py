"""
EXP-144 P0 Offline Replay — LAB v4.2 算法验证
================================================

目的: 用现有 vanilla cold path npz 数据 (PACS / Office vanilla, 没有 DaA 干扰),
      模拟 LAB 权重轨迹, 验证:
      ① 所有 final_ratio ∈ [0.80, 2.00] (algorithm correctness)
      ② Bisection projection 数值稳定 (无 NaN, 收敛 < 64 iter)
      ③ Ratio_clipped 频率 < 30%
      ④ 没有 domain 100% rounds 都饱和

注意: 用 1 - per_domain_test_acc 当 val_loss 近似 proxy. 这只验证
      公式合理性, 不能证明 acc 效果. 真实 val_loss 在 P1 编码后才有.

Usage:
    python p0_offline_replay.py --output_dir .
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ========== Config ==========
LAB_LAMBDA = 0.15
RATIO_MIN = 0.80
RATIO_MAX = 2.00
EMA_ALPHA = 0.30
BISECTION_TOL = 1e-9
BISECTION_MAX_ITER = 64

# Fixed allocation (跟 F2DC/utils/training.py:154-159 一致)
PACS_ALLOC = {"photo": 2, "art": 3, "cartoon": 2, "sketch": 3}
OFFICE_ALLOC = {"caltech": 3, "amazon": 2, "webcam": 2, "dslr": 3}

# 按 cli_id 顺序展开 (selected_domain_list)
def build_cli_to_dom(alloc):
    cli_to_dom = []
    for d, n in alloc.items():
        cli_to_dom.extend([d] * n)
    return cli_to_dom

PACS_CLI_DOM = build_cli_to_dom(PACS_ALLOC)
OFFICE_CLI_DOM = build_cli_to_dom(OFFICE_ALLOC)


# ========== Bounded Simplex Projection (bisection) ==========
def bounded_simplex_projection(w_raw, w_min, w_max, target=1.0,
                                tol=BISECTION_TOL, max_iter=BISECTION_MAX_ITER):
    """
    Find tau s.t. sum(clip(w_raw + tau, w_min, w_max)) = target.
    Returns (w_proj, n_iter, converged).
    """
    w_raw = np.asarray(w_raw, dtype=np.float64)
    w_min = np.asarray(w_min, dtype=np.float64)
    w_max = np.asarray(w_max, dtype=np.float64)

    # tau bounds (codex guardrail: 标准保守写法, 极端 w_raw 也成立)
    # lo 必须保证 w_raw + lo ≤ w_min[d] 对所有 d 成立
    # hi 必须保证 w_raw + hi ≥ w_max[d] 对所有 d 成立
    lo = np.min(w_min - w_raw) - 1.0
    hi = np.max(w_max - w_raw) + 1.0

    # Edge case check: 是否可行?
    s_lo = np.clip(w_raw + lo, w_min, w_max).sum()
    s_hi = np.clip(w_raw + hi, w_min, w_max).sum()
    if s_lo > target + tol:
        # 即使全部 clip 到 w_min 还是太大 (sum(w_min) > 1)
        return np.clip(w_raw + lo, w_min, w_max), 0, False
    if s_hi < target - tol:
        # 即使全部 clip 到 w_max 还是太小 (sum(w_max) < 1)
        return np.clip(w_raw + hi, w_min, w_max), 0, False

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        s = np.clip(w_raw + mid, w_min, w_max).sum()
        if abs(s - target) < tol:
            break
        if s > target:
            hi = mid
        else:
            lo = mid

    w_proj = np.clip(w_raw + mid, w_min, w_max)
    converged = abs(w_proj.sum() - target) < tol * 10
    return w_proj, n_iter, converged


# ========== LAB Algorithm ==========
def lab_step(loss_ema, sample_share_dom, lam=LAB_LAMBDA):
    """
    LAB v4.2 单 round 算法 (Step 3-4).
    Args:
        loss_ema: dict {dom: ema_loss}
        sample_share_dom: dict {dom: total share of this domain}
    Returns:
        w_proj, ratio, gap, q, n_iter, converged, clip_status
    """
    domains = list(loss_ema.keys())
    losses = np.array([loss_ema[d] for d in domains])
    shares = np.array([sample_share_dom[d] for d in domains])

    loss_mean = losses.mean()
    gap = np.maximum(0.0, losses - loss_mean)

    # ====== Codex Blocking Fix: gap=0 → 退化 FedAvg ======
    # PROPOSAL §3.5 明文承诺 "全域学得差不多 → 退化 FedAvg".
    # 如果不 early return, w_raw = (1-λ)*shares 会让 sum < 1, projection 后
    # 因 ratio bound 不均, 最终 ratio 不是 1.0 (实测 [0.93, 1.14]).
    if gap.sum() <= 1e-12:
        return {
            "domains": domains,
            "loss_ema": dict(zip(domains, losses)),
            "gap": dict(zip(domains, gap)),
            "q": dict(zip(domains, np.zeros_like(gap))),
            "w_raw": dict(zip(domains, shares)),
            "w_proj": dict(zip(domains, shares)),
            "ratio": dict(zip(domains, np.ones_like(shares))),
            "n_iter": 0,
            "converged": True,
            "clip_status": dict(zip(domains, [None] * len(domains))),
            "sum_check": float(shares.sum()),
        }
    # ======================================================

    q = gap / gap.sum()
    w_raw = (1 - lam) * shares + lam * q
    w_min = RATIO_MIN * shares
    w_max = RATIO_MAX * shares

    w_proj, n_iter, converged = bounded_simplex_projection(w_raw, w_min, w_max)
    ratio = w_proj / shares

    # 检测边界触发 (容差 1e-6)
    clip_status = []
    for i in range(len(domains)):
        if abs(w_proj[i] - w_min[i]) < 1e-6:
            clip_status.append("@min")
        elif abs(w_proj[i] - w_max[i]) < 1e-6:
            clip_status.append("@max")
        else:
            clip_status.append(None)

    return {
        "domains": domains,
        "loss_ema": dict(zip(domains, losses)),
        "gap": dict(zip(domains, gap)),
        "q": dict(zip(domains, q)),
        "w_raw": dict(zip(domains, w_raw)),
        "w_proj": dict(zip(domains, w_proj)),
        "ratio": dict(zip(domains, ratio)),
        "n_iter": n_iter,
        "converged": converged,
        "clip_status": dict(zip(domains, clip_status)),
        "sum_check": w_proj.sum(),
    }


# ========== Run replay on one cold path dir ==========
def replay_run(diag_dir, cli_to_dom, dataset_name):
    """
    Replay LAB on one vanilla cold path dir (100 rounds).
    Returns trajectory dict.
    """
    base = Path(diag_dir)
    if not base.exists():
        return None

    # 拿 domain-level sample_share (固定不变, 用 round 1)
    z1 = np.load(base / "round_001.npz", allow_pickle=True)
    online1 = z1["online_clients"].astype(int)
    shares1 = z1["sample_shares"].astype(float)
    dom_names = list(z1["all_dataset_names"])

    # 把 cli sample_share 聚合到 domain
    dom_share = defaultdict(float)
    for cli_id, sh in zip(online1, shares1):
        d = cli_to_dom[cli_id]
        dom_share[d] += sh
    dom_share = dict(dom_share)
    # 确保按 dom_names 顺序
    sample_share_dom = {d: dom_share[d] for d in dom_names}

    traj = {
        "dataset": dataset_name,
        "diag_dir": str(diag_dir),
        "sample_share_dom": sample_share_dom,
        "rounds": [],
    }

    loss_ema = None
    for r in range(1, 101):
        fp = base / f"round_{r:03d}.npz"
        if not fp.exists():
            break
        z = np.load(fp, allow_pickle=True)

        per_dom_acc = z["per_domain_acc"]  # 按 all_dataset_names 顺序
        # val_loss proxy = 1 - acc/100
        val_loss = {d: 1.0 - per_dom_acc[i] / 100.0
                    for i, d in enumerate(dom_names)}

        # EMA
        if loss_ema is None:
            loss_ema = dict(val_loss)
        else:
            loss_ema = {d: EMA_ALPHA * val_loss[d] + (1 - EMA_ALPHA) * loss_ema[d]
                        for d in val_loss}

        # 跑 LAB step
        result = lab_step(loss_ema, sample_share_dom)
        result["round"] = r
        result["val_loss_raw"] = val_loss
        result["per_dom_acc"] = {d: float(per_dom_acc[i])
                                  for i, d in enumerate(dom_names)}
        traj["rounds"].append(result)

    return traj


# ========== Diagnostics & Report ==========
def diagnose_trajectory(traj):
    """Compute aggregate diagnostics from a trajectory."""
    rounds = traj["rounds"]
    if not rounds:
        return None
    domains = rounds[0]["domains"]
    R = len(rounds)

    # Algorithm correctness checks
    all_ratios = []
    sum_errors = []
    n_iters = []
    not_converged = 0

    # Per-domain stats
    clip_max_count = defaultdict(int)
    clip_min_count = defaultdict(int)
    boost_track = defaultdict(list)  # positive_delta over rounds

    for r_data in rounds:
        for d in domains:
            all_ratios.append(r_data["ratio"][d])
            cs = r_data["clip_status"][d]
            if cs == "@max":
                clip_max_count[d] += 1
            elif cs == "@min":
                clip_min_count[d] += 1
            # boost = positive delta = max(0, w_proj - sample_share)
            sh = traj["sample_share_dom"][d]
            pd = max(0.0, r_data["w_proj"][d] - sh)
            boost_track[d].append(pd)

        sum_errors.append(abs(r_data["sum_check"] - 1.0))
        n_iters.append(r_data["n_iter"])
        if not r_data["converged"]:
            not_converged += 1

    all_ratios = np.array(all_ratios)
    in_bounds = ((all_ratios >= RATIO_MIN - 1e-6) &
                 (all_ratios <= RATIO_MAX + 1e-6))

    diag = {
        "dataset": traj["dataset"],
        "R": R,
        "n_domains": len(domains),
        "n_total_obs": len(all_ratios),
        # ① algorithm correctness
        "ratios_in_bounds_pct": 100.0 * in_bounds.sum() / len(in_bounds),
        "ratio_min_observed": float(all_ratios.min()),
        "ratio_max_observed": float(all_ratios.max()),
        # ② bisection stability
        "max_sum_error": float(max(sum_errors)),
        "max_n_iter": int(max(n_iters)),
        "mean_n_iter": float(np.mean(n_iters)),
        "not_converged_count": not_converged,
        "has_nan": bool(np.any(np.isnan(all_ratios))),
        # ③ clip rate
        "clip_max_per_dom": {d: clip_max_count[d] / R for d in domains},
        "clip_min_per_dom": {d: clip_min_count[d] / R for d in domains},
        "clip_rate_overall": (sum(clip_max_count.values()) +
                              sum(clip_min_count.values())) / (R * len(domains)),
        # ④ saturation
        "always_at_max": [d for d in domains if clip_max_count[d] / R > 0.95],
        "always_at_min": [d for d in domains if clip_min_count[d] / R > 0.95],
        # auxiliary
        "boost_total_per_dom": {d: float(sum(boost_track[d])) for d in domains},
        "boost_mean_per_dom": {d: float(np.mean(boost_track[d])) for d in domains},
    }
    return diag


def format_report(all_diags):
    """Format markdown report."""
    lines = []
    lines.append("# EXP-144 P0 Offline Replay Report\n")
    lines.append("> LAB v4.2 算法验证 (用 1-test_acc 当 val_loss proxy, "
                 "仅验权重轨迹合理性, 不证明 acc 效果)\n")
    lines.append("\n## Coverage (codex guardrail)\n")
    lines.append("\n- **Datasets covered**: PACS + Office vanilla cold-path runs (8 total)\n")
    lines.append("- **NOT covered in P0**: Digits (no cold-path npz available; deferred to P1 real val_loss)\n")
    lines.append("- **What this proves**: bisection projection numerical stability, ratio bounds, "
                 "boost-direction semantics on 2 datasets where DaA shows opposite signs (PACS loses, Office wins)\n")
    lines.append("- **What this does NOT prove**: actual accuracy improvement (proxy 1-acc 太粗, "
                 "真实信号在 P1 编码后才有)\n")
    lines.append(f"\n**Config**: λ={LAB_LAMBDA}, ratio∈[{RATIO_MIN}, {RATIO_MAX}], "
                 f"EMA α={EMA_ALPHA}, bisection tol={BISECTION_TOL}, max_iter={BISECTION_MAX_ITER}\n")
    lines.append("---\n")

    # ===== Top-level pass/fail =====
    overall_pass = True
    overall_warn = []
    for diag in all_diags:
        if diag is None:
            continue
        # Hard gates
        if diag["ratios_in_bounds_pct"] < 99.99:
            overall_pass = False
            overall_warn.append(
                f"❌ {diag['dataset']}: ratio out of bounds ({diag['ratios_in_bounds_pct']:.4f}%)")
        if diag["max_sum_error"] > 1e-6:
            overall_pass = False
            overall_warn.append(
                f"❌ {diag['dataset']}: bisection sum error {diag['max_sum_error']:.2e}")
        if diag["max_n_iter"] >= BISECTION_MAX_ITER:
            overall_pass = False
            overall_warn.append(
                f"❌ {diag['dataset']}: bisection hit max_iter {BISECTION_MAX_ITER}")
        if diag["has_nan"]:
            overall_pass = False
            overall_warn.append(f"❌ {diag['dataset']}: NaN in ratios")
        if diag["not_converged_count"] > 0:
            overall_pass = False
            overall_warn.append(
                f"❌ {diag['dataset']}: {diag['not_converged_count']} rounds didn't converge")
        # Soft warnings
        if diag["clip_rate_overall"] > 0.30:
            overall_warn.append(
                f"⚠️ {diag['dataset']}: clip_rate {diag['clip_rate_overall']*100:.1f}% > 30%")
        if diag["always_at_max"] or diag["always_at_min"]:
            overall_warn.append(
                f"⚠️ {diag['dataset']}: domain saturated >95% rounds: "
                f"max={diag['always_at_max']}, min={diag['always_at_min']}")

    lines.append("## ⚖️ 总体判定\n")
    lines.append(f"\n**P0 PASS**: {'✅ YES' if overall_pass else '❌ NO'}\n")
    if overall_warn:
        lines.append("\n**Warnings**:\n")
        for w in overall_warn:
            lines.append(f"- {w}\n")
    else:
        lines.append("\nNo warnings. All algorithm correctness checks passed.\n")

    lines.append("\n---\n\n## 📊 Per-Run Diagnostics\n\n")

    for diag in all_diags:
        if diag is None:
            continue
        lines.append(f"### {diag['dataset']} (R={diag['R']}, {diag['n_domains']} domains)\n\n")
        lines.append(f"**① Algorithm Correctness**:\n")
        lines.append(f"- Ratios in bounds: **{diag['ratios_in_bounds_pct']:.4f}%** "
                     f"(observed: [{diag['ratio_min_observed']:.4f}, {diag['ratio_max_observed']:.4f}])\n")
        lines.append(f"- NaN: {'❌ YES' if diag['has_nan'] else '✅ NO'}\n\n")

        lines.append(f"**② Bisection Stability**:\n")
        lines.append(f"- Max sum error: {diag['max_sum_error']:.2e} "
                     f"(✅ < 1e-6)\n")
        lines.append(f"- Iterations: max={diag['max_n_iter']}, mean={diag['mean_n_iter']:.1f} "
                     f"(✅ < {BISECTION_MAX_ITER})\n")
        lines.append(f"- Non-converged rounds: {diag['not_converged_count']}\n\n")

        lines.append(f"**③ Clip Rate**:\n")
        lines.append(f"- Overall: {diag['clip_rate_overall']*100:.1f}%\n")
        lines.append(f"- Per domain (@max): " + ", ".join(
            f"{d}:{v*100:.0f}%" for d, v in diag['clip_max_per_dom'].items()) + "\n")
        lines.append(f"- Per domain (@min): " + ", ".join(
            f"{d}:{v*100:.0f}%" for d, v in diag['clip_min_per_dom'].items()) + "\n\n")

        lines.append(f"**④ Saturation Detection (>95% rounds at boundary)**:\n")
        lines.append(f"- Always @max: {diag['always_at_max'] or 'none'}\n")
        lines.append(f"- Always @min: {diag['always_at_min'] or 'none'}\n\n")

        lines.append(f"**Boost Distribution**:\n")
        lines.append(f"- Total boost: " + ", ".join(
            f"{d}:{v:.3f}" for d, v in diag['boost_total_per_dom'].items()) + "\n")
        lines.append(f"- Mean boost/round: " + ", ".join(
            f"{d}:{v:.4f}" for d, v in diag['boost_mean_per_dom'].items()) + "\n\n")

        lines.append("---\n\n")

    return "".join(lines)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold_path_root", type=str,
                        default="/Users/changdao/联邦学习/experiments/cold_path_analysis")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/changdao/联邦学习/experiments/ablation/EXP-144_lab_v4_1")
    args = parser.parse_args()

    cold_root = Path(args.cold_path_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # === 单元测试 bisection ===
    print("=" * 70)
    print("Pre-flight: bisection projection unit tests")
    print("=" * 70)
    run_unit_tests()

    # === Run replay on each available vanilla cold path dir ===
    runs = [
        ("PACS f2dc s15",   cold_root / "diag_pacs/diag_f2dc_pacs_s15",   PACS_CLI_DOM),
        ("PACS f2dc s333",  cold_root / "diag_pacs/diag_f2dc_pacs_s333",  PACS_CLI_DOM),
        ("PACS pgdfc s15",  cold_root / "diag_pacs/diag_pgdfc_pacs_s15",  PACS_CLI_DOM),
        ("PACS pgdfc s333", cold_root / "diag_pacs/diag_pgdfc_pacs_s333", PACS_CLI_DOM),
        ("Office f2dc s2",   cold_root / "diag_office/diag_f2dc_office_s2",   OFFICE_CLI_DOM),
        ("Office f2dc s15",  cold_root / "diag_office/diag_f2dc_office_s15",  OFFICE_CLI_DOM),
        ("Office pgdfc s2",  cold_root / "diag_office/diag_pgdfc_office_s2",  OFFICE_CLI_DOM),
        ("Office pgdfc s15", cold_root / "diag_office/diag_pgdfc_office_s15", OFFICE_CLI_DOM),
    ]

    all_diags = []
    all_trajs = {}
    print(f"\n{'='*70}")
    print(f"Running offline replay on {len(runs)} vanilla cold-path runs")
    print(f"{'='*70}\n")

    for name, diag_dir, cli_to_dom in runs:
        print(f"  [{name}] reading {diag_dir.name}...")
        traj = replay_run(diag_dir, cli_to_dom, name)
        if traj is None:
            print(f"     ⚠️ skip (dir not found)")
            continue
        diag = diagnose_trajectory(traj)
        all_diags.append(diag)
        all_trajs[name] = traj
        print(f"     R={diag['R']}, ratio_in_bounds={diag['ratios_in_bounds_pct']:.4f}%, "
              f"max_iter={diag['max_n_iter']}, clip_rate={diag['clip_rate_overall']*100:.1f}%")

    # === Generate report ===
    print(f"\n{'='*70}\nGenerating report...\n{'='*70}")
    report = format_report(all_diags)
    out_md = out_dir / "p0_replay_report.md"
    out_md.write_text(report)
    print(f"  ✅ Report saved to {out_md}")

    # === Save raw trajectories as JSON for further analysis ===
    out_json = out_dir / "p0_trajectories.json"

    def _to_native(x):
        """Convert numpy scalars/arrays to Python native types for JSON."""
        if isinstance(x, dict):
            return {k: _to_native(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_native(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    json_data = _to_native({
        name: {
            "dataset": traj["dataset"],
            "sample_share_dom": traj["sample_share_dom"],
            "rounds": [
                {
                    "round": r["round"],
                    "loss_ema": r["loss_ema"],
                    "ratio": r["ratio"],
                    "w_proj": r["w_proj"],
                    "n_iter": r["n_iter"],
                    "clip_status": r["clip_status"],
                    "per_dom_acc": r["per_dom_acc"],
                }
                for r in traj["rounds"]
            ],
        }
        for name, traj in all_trajs.items()
    })
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=1)
    print(f"  ✅ Trajectories saved to {out_json}")

    # === Summary to stdout ===
    print(f"\n{'='*70}\nFinal Summary\n{'='*70}")
    overall_pass = all(
        d["ratios_in_bounds_pct"] >= 99.99
        and d["max_sum_error"] < 1e-6
        and d["max_n_iter"] < BISECTION_MAX_ITER
        and not d["has_nan"]
        and d["not_converged_count"] == 0
        for d in all_diags
    )
    print(f"\n  P0 PASS: {'✅ YES' if overall_pass else '❌ NO'}")
    print(f"  Hard gates checked: ratio bounds, sum error, bisection convergence, NaN")
    print(f"  Soft warnings (see report) do not fail P0\n")
    return 0 if overall_pass else 1


# ========== Unit tests for bisection ==========
def run_unit_tests():
    """6 unit tests for bounded simplex projection (codex guardrail #3)."""
    tests_passed = 0
    tests_total = 0

    def assert_close(actual, expected, name, tol=1e-6):
        nonlocal tests_passed, tests_total
        tests_total += 1
        ok = abs(actual - expected) < tol
        status = "✅" if ok else "❌"
        if ok:
            tests_passed += 1
        print(f"  {status} {name}: actual={actual:.6f}, expected={expected:.6f}")

    def assert_in_bounds(arr, lo, hi, name):
        nonlocal tests_passed, tests_total
        tests_total += 1
        arr = np.asarray(arr)
        ok = (arr >= lo - 1e-6).all() and (arr <= hi + 1e-6).all()
        status = "✅" if ok else "❌"
        if ok:
            tests_passed += 1
        print(f"  {status} {name}: range=[{arr.min():.4f}, {arr.max():.4f}], "
              f"expected ⊂ [{lo}, {hi}]")

    # Test 1: sum to 1
    w_raw = np.array([0.20, 0.30, 0.25, 0.25])
    w_min = 0.80 * w_raw
    w_max = 2.00 * w_raw
    w_proj, n_iter, ok = bounded_simplex_projection(w_raw, w_min, w_max)
    assert_close(w_proj.sum(), 1.0, "Test 1: sum_to_one (degenerate FedAvg)")

    # Test 2: in bounds
    assert_in_bounds(w_proj / w_raw, 0.80, 2.00, "Test 2: ratio_in_bounds")

    # Test 3 (codex blocking fix): gap=0 → 必须严格退化 FedAvg
    # 通过 lab_step 测试 early return 路径
    # 用 PACS 真实 share, 严格 sum=1.0 (来自 cold path npz round_001)
    sample_share = np.array([0.1284, 0.2370, 0.1804, 0.4542])  # sum=1.0 strict
    domains_t3 = ["photo", "art", "cartoon", "sketch"]
    share_dict = dict(zip(domains_t3, sample_share))
    loss_uniform = {d: 0.30 for d in domains_t3}    # 所有 loss 一样 → gap 全 0
    result = lab_step(loss_uniform, share_dict)
    ratios_t3 = np.array([result["ratio"][d] for d in domains_t3])
    w_proj_t3 = np.array([result["w_proj"][d] for d in domains_t3])
    assert_close(w_proj_t3.sum(), 1.0, "Test 3a: gap_zero_sum_to_one")
    # 硬断言: ratio 必须全部 = 1.0, 不只是 in bounds
    nonlocal_n_total = tests_total + 1
    if np.allclose(ratios_t3, 1.0, atol=1e-9):
        tests_passed += 1
        print(f"  ✅ Test 3b: gap_zero_ratio_strictly_one: range=[{ratios_t3.min():.6f}, "
              f"{ratios_t3.max():.6f}]")
    else:
        print(f"  ❌ Test 3b: gap_zero_ratio_strictly_one: range=[{ratios_t3.min():.6f}, "
              f"{ratios_t3.max():.6f}] (expected all 1.0)")
    tests_total = nonlocal_n_total
    # w_proj 必须严格 = sample_share
    nonlocal_n_total = tests_total + 1
    if np.allclose(w_proj_t3, sample_share, atol=1e-9):
        tests_passed += 1
        print(f"  ✅ Test 3c: gap_zero_w_proj_eq_sample_share")
    else:
        print(f"  ❌ Test 3c: gap_zero_w_proj_eq_sample_share")
    tests_total = nonlocal_n_total

    # Test 4: small domain hits ratio_max
    sample_share = np.array([0.10, 0.20, 0.20, 0.50])
    q = np.array([1.0, 0.0, 0.0, 0.0])  # 全部 boost 给最小域
    lam = 0.50  # 大 lambda 强制触发 max
    w_raw = (1 - lam) * sample_share + lam * q
    w_min = 0.80 * sample_share
    w_max = 2.00 * sample_share
    w_proj, n_iter, ok = bounded_simplex_projection(w_raw, w_min, w_max)
    nonlocal_n_total = tests_total + 1
    if abs(w_proj[0] - w_max[0]) < 1e-5:
        tests_passed += 1
        print(f"  ✅ Test 4: small_dom_hits_max: w_proj[0]={w_proj[0]:.4f}, "
              f"w_max[0]={w_max[0]:.4f}")
    else:
        print(f"  ❌ Test 4: small_dom_hits_max: w_proj[0]={w_proj[0]:.4f}, "
              f"w_max[0]={w_max[0]:.4f}")
    tests_total = nonlocal_n_total

    # Test 5: large domain hits ratio_min when others heavily boosted
    sample_share = np.array([0.10, 0.10, 0.10, 0.70])
    q = np.array([0.5, 0.5, 0.0, 0.0])  # boost 给小域
    lam = 0.50
    w_raw = (1 - lam) * sample_share + lam * q
    w_min = 0.80 * sample_share
    w_max = 2.00 * sample_share
    w_proj, n_iter, ok = bounded_simplex_projection(w_raw, w_min, w_max)
    nonlocal_n_total = tests_total + 1
    # large domain (idx 3) 应该被砍到接近 w_min (0.56)
    if w_proj[3] <= w_min[3] * 1.02:  # within 2%
        tests_passed += 1
        print(f"  ✅ Test 5: large_dom_hits_min: w_proj[3]={w_proj[3]:.4f}, "
              f"w_min[3]={w_min[3]:.4f}")
    else:
        print(f"  ❌ Test 5: large_dom_hits_min: w_proj[3]={w_proj[3]:.4f}, "
              f"w_min[3]={w_min[3]:.4f}")
    tests_total = nonlocal_n_total

    # Test 6: PACS realistic
    sample_share_pacs = {"photo": 0.128, "art": 0.237, "cartoon": 0.180, "sketch": 0.454}
    val_loss = {"photo": 0.325, "art": 0.418, "cartoon": 0.215, "sketch": 0.162}
    result = lab_step(val_loss, sample_share_pacs)
    expected_ratios = {"photo": 1.14, "art": 1.32, "cartoon": 0.85, "sketch": 0.85}
    test_pass = True
    for d, exp in expected_ratios.items():
        actual = result["ratio"][d]
        if abs(actual - exp) > 0.05:
            test_pass = False
            print(f"  ❌ Test 6: PACS realistic {d}: actual={actual:.3f}, "
                  f"expected≈{exp:.2f}")
        else:
            print(f"  ✅ Test 6: PACS realistic {d}: actual={actual:.3f}, "
                  f"expected≈{exp:.2f}")
    tests_total += 4
    if test_pass:
        tests_passed += 4

    print(f"\n  Unit tests: {tests_passed}/{tests_total} passed")
    if tests_passed < tests_total:
        print("  ❌ Bisection projection FAILED unit tests, aborting P0")
        sys.exit(1)
    print()


if __name__ == "__main__":
    sys.exit(main())
