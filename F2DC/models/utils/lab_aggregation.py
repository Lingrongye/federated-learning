"""
LAB v4.2 (Loss-Aware Boost) Aggregation — Pure Algorithm Module
==================================================================

本文件: 不依赖任何 F2DC 训练代码, 纯 numpy 实现, 可被任何 model 引入.

核心:
- bounded_simplex_projection: bisection 求 tau 使 sum(clip(w_raw+tau, w_min, w_max)) = 1
- lab_step: 5 步算法 (gap=0 时早 return FedAvg, 满足 PROPOSAL §3.5)
- LabState: 持有 EMA loss + cumulative boost + waste detection 状态

不在本文件做的事:
- 不读 cold path npz (P0 已验证, 这里假设直接喂 val_loss dict)
- 不写 dataset partition (在 datasets/utils/lab_partition.py)
- 不写 F2DC trainer (在 models/f2dc_pg_lab.py)

跟 P0 offline replay (experiments/ablation/EXP-144_lab_v4_1/p0_offline_replay.py)
完全等价, 公式从 P0 拷贝过来确保 algorithm correctness 一致.

PROPOSAL: experiments/ablation/EXP-144_lab_v4_1/PROPOSAL.md
P0 报告: experiments/ablation/EXP-144_lab_v4_1/p0_replay_report.md (已 PASS)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Bounded Simplex Projection (codex guardrail #2)
# =============================================================================
def bounded_simplex_projection(
    w_raw: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    target: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 64,
) -> Tuple[np.ndarray, int, bool]:
    """
    Find tau s.t. sum(clip(w_raw + tau, w_min, w_max)) = target via bisection.

    Returns:
        (w_proj, n_iter, converged)
    """
    w_raw = np.asarray(w_raw, dtype=np.float64)
    w_min = np.asarray(w_min, dtype=np.float64)
    w_max = np.asarray(w_max, dtype=np.float64)

    # tau bounds (codex guardrail: 标准保守写法, 极端 w_raw 也成立)
    # lo 必须保证 w_raw + lo ≤ w_min[d] 对所有 d 成立
    # hi 必须保证 w_raw + hi ≥ w_max[d] 对所有 d 成立
    lo = float(np.min(w_min - w_raw)) - 1.0
    hi = float(np.max(w_max - w_raw)) + 1.0

    # Edge case: 不可行
    s_lo = np.clip(w_raw + lo, w_min, w_max).sum()
    s_hi = np.clip(w_raw + hi, w_min, w_max).sum()
    if s_lo > target + tol:
        return np.clip(w_raw + lo, w_min, w_max), 0, False
    if s_hi < target - tol:
        return np.clip(w_raw + hi, w_min, w_max), 0, False

    n_iter = 0
    mid = 0.5 * (lo + hi)
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
    converged = abs(w_proj.sum() - target) < tol * 10.0
    return w_proj, n_iter, converged


# =============================================================================
# LAB Step (PROPOSAL §3.5)
# =============================================================================
def lab_step(
    loss_ema: Dict[str, float],
    sample_share_dom: Dict[str, float],
    lam: float = 0.15,
    ratio_min: float = 0.80,
    ratio_max: float = 2.00,
    bisection_tol: float = 1e-9,
    bisection_max_iter: int = 64,
) -> Dict[str, Any]:
    """
    LAB v4.2 单 round 主公式 (Step 3-5 of PROPOSAL §3.5).

    Args:
        loss_ema: {dom_name: ema_smoothed_val_loss}
        sample_share_dom: {dom_name: total_sample_share_in_this_domain},
                          sum to 1.0 over all domains
        lam: λ-mix coefficient (PROPOSAL default 0.15)
        ratio_min, ratio_max: bounded simplex constraint
        bisection_tol, bisection_max_iter: projection numerics

    Returns: dict with 11 keys (用于诊断 dump):
        - domains, loss_ema, gap, q, w_raw, w_proj, ratio
        - n_iter, converged, clip_status, sum_check
    """
    domains = list(loss_ema.keys())
    losses = np.array([loss_ema[d] for d in domains], dtype=np.float64)
    shares = np.array([sample_share_dom[d] for d in domains], dtype=np.float64)

    loss_mean = losses.mean()
    gap = np.maximum(0.0, losses - loss_mean)

    # ===== Codex Blocking Fix: gap=0 → 严格退化 FedAvg =====
    # PROPOSAL §3.5 明文承诺. 不 early return 会导致 ratio 不是 1.0.
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
    # ========================================================

    q = gap / gap.sum()
    w_raw = (1.0 - lam) * shares + lam * q
    w_min = ratio_min * shares
    w_max = ratio_max * shares

    w_proj, n_iter, converged = bounded_simplex_projection(
        w_raw, w_min, w_max,
        target=1.0, tol=bisection_tol, max_iter=bisection_max_iter,
    )
    ratio = w_proj / shares

    # 检测边界触发 (容差 1e-6)
    clip_status: Dict[str, Optional[str]] = {}
    for i, d in enumerate(domains):
        if abs(w_proj[i] - w_min[i]) < 1e-6:
            clip_status[d] = "@min"
        elif abs(w_proj[i] - w_max[i]) < 1e-6:
            clip_status[d] = "@max"
        else:
            clip_status[d] = None

    return {
        "domains": domains,
        "loss_ema": dict(zip(domains, losses)),
        "gap": dict(zip(domains, gap)),
        "q": dict(zip(domains, q)),
        "w_raw": dict(zip(domains, w_raw)),
        "w_proj": dict(zip(domains, w_proj)),
        "ratio": dict(zip(domains, ratio)),
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "clip_status": clip_status,
        "sum_check": float(w_proj.sum()),
    }


# =============================================================================
# LAB State (cross-round EMA + cumulative boost + waste detection)
# =============================================================================
class LabState:
    """
    持有 LAB 跨 round 状态: val_loss EMA, cumulative positive_delta (boost),
    per-domain acc trajectory, ROI / waste detection.

    一次实验只 instantiate 一次 (在 model.__init__ 里).

    用法:
        # round r 末尾
        state.update_from_eval(round_r, per_dom_acc)        # test acc, sanity log
        state.update_val_loss(round_r, val_loss_per_dom)    # 真信号
        result = state.compute_lab(round_r, sample_share_dom)
        state.update_boost_history(round_r, result)
        diag = state.full_diagnostic(round_r, result)        # 20 字段, dump 用
    """

    def __init__(
        self,
        lam: float = 0.15,
        ratio_min: float = 0.80,
        ratio_max: float = 2.00,
        ema_alpha: float = 0.30,
        window_size: int = 20,
        waste_roi_threshold: float = 0.5,
    ):
        self.lam = float(lam)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.ema_alpha = float(ema_alpha)
        self.window_size = int(window_size)
        self.waste_roi_threshold = float(waste_roi_threshold)

        # 跨 round 状态
        self.val_loss_ema: Dict[str, float] = {}
        self.val_loss_raw_history: Dict[str, List[float]] = defaultdict(list)
        self.acc_history: Dict[str, List[float]] = defaultdict(list)
        self.boost_history: Dict[str, List[float]] = defaultdict(list)  # positive_delta
        self.ratio_history: Dict[str, List[float]] = defaultdict(list)
        self.clip_at_max_count: Dict[str, int] = defaultdict(int)
        self.clip_at_min_count: Dict[str, int] = defaultdict(int)
        self.last_round_seen: int = 0

        # 是否曾经收到过 val_loss (R=1 时还没有, 走 FedAvg)
        self.has_val_loss: bool = False

    # ------------------------------------------------------------
    # 信号更新
    # ------------------------------------------------------------
    def update_val_loss(self, round_idx: int, val_loss_per_dom: Dict[str, float]) -> None:
        """每 round 聚合后, server 用 client 上传的 (loss_sum, val_n) 算出 per-dom val_loss,
        喂给本函数. EMA 平滑后存入 val_loss_ema, 给下一 round LAB 用.

        codex blocking fix: 空 dict 不 set has_val_loss=True (否则 round 0 末
        val 全空时会把 has_val_loss 错置 True, 下一轮 compute_lab 缺 domain KeyError).
        """
        if not val_loss_per_dom:
            # 空 dict (val 全部 cli 都没上传, 比如 ImageFolder_Custom 不被识别时)
            # 不更新 has_val_loss, 下一轮自动走 FedAvg fallback
            return
        for d, v in val_loss_per_dom.items():
            v = float(v)
            self.val_loss_raw_history[d].append(v)
            if d not in self.val_loss_ema:
                self.val_loss_ema[d] = v
            else:
                self.val_loss_ema[d] = (
                    self.ema_alpha * v + (1.0 - self.ema_alpha) * self.val_loss_ema[d]
                )
        self.has_val_loss = True
        self.last_round_seen = max(self.last_round_seen, round_idx)

    def update_test_acc(self, round_idx: int, acc_per_dom: Dict[str, float]) -> None:
        """test acc 仅用于 sanity log + ROI 计算 (NOT for LAB signal)."""
        for d, a in acc_per_dom.items():
            self.acc_history[d].append(float(a))

    # ------------------------------------------------------------
    # LAB 计算
    # ------------------------------------------------------------
    def compute_lab(
        self,
        round_idx: int,
        sample_share_dom: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        round r 开始时调用. 用 val_loss_ema (来自 r-1 聚合后测的) 算 LAB 权重.

        如果还没有 val_loss (R=1), 退化为 FedAvg (ratio 全 1.0).
        """
        domains = list(sample_share_dom.keys())
        shares = np.array([sample_share_dom[d] for d in domains])

        def _fallback(reason: str):
            return {
                "domains": domains,
                "loss_ema": {d: 0.0 for d in domains},
                "gap": {d: 0.0 for d in domains},
                "q": {d: 0.0 for d in domains},
                "w_raw": dict(zip(domains, shares)),
                "w_proj": dict(zip(domains, shares)),
                "ratio": {d: 1.0 for d in domains},
                "n_iter": 0,
                "converged": True,
                "clip_status": {d: None for d in domains},
                "sum_check": float(shares.sum()),
                "fallback_to_fedavg": True,
                "fallback_reason": reason,
            }

        if not self.has_val_loss:
            return _fallback("round_1_no_val_loss_yet")

        # codex blocking fix: 缺 domain 时走 fallback, 不 KeyError
        missing = [d for d in domains if d not in self.val_loss_ema]
        if missing:
            return _fallback(f"missing_val_loss_for_domains:{missing}")

        # 喂给 lab_step
        loss_ema_for_step = {d: self.val_loss_ema[d] for d in domains}
        result = lab_step(
            loss_ema=loss_ema_for_step,
            sample_share_dom=sample_share_dom,
            lam=self.lam,
            ratio_min=self.ratio_min,
            ratio_max=self.ratio_max,
        )
        result["fallback_to_fedavg"] = False
        result["fallback_reason"] = None
        return result

    def update_boost_record(
        self,
        round_idx: int,
        sample_share_dom: Dict[str, float],
        result: Dict[str, Any],
    ) -> Dict[str, float]:
        """记录本 round positive_delta + ratio + clip_status (一站式).

        被 model.loc_update 在每 round 末调用. 顺带把 clip_count 增量补上
        (修 codex Important #5: clip count 之前只在 update_boost_history 增, 没人调).

        Returns:
            positive_delta: {dom: max(0, w_proj - sample_share)}
        """
        positive_delta: Dict[str, float] = {}
        for d in result["domains"]:
            sh = float(sample_share_dom[d])
            wp = float(result["w_proj"][d])
            pd = max(0.0, wp - sh)
            positive_delta[d] = pd
            self.boost_history[d].append(pd)
            # 同步记 ratio + clip count (修 codex Important #5)
            self.ratio_history[d].append(float(result["ratio"][d]))
            cs = result["clip_status"].get(d)
            if cs == "@max":
                self.clip_at_max_count[d] += 1
            elif cs == "@min":
                self.clip_at_min_count[d] += 1
        return positive_delta

    # ------------------------------------------------------------
    # 诊断指标 (20+ 字段, 给 diagnostic.py dump 用)
    # ------------------------------------------------------------
    def full_diagnostic(
        self,
        round_idx: int,
        sample_share_dom: Dict[str, float],
        lab_result: Dict[str, Any],
        positive_delta: Dict[str, float],
        val_n_per_dom: Optional[Dict[str, int]] = None,
        val_class_counts: Optional[Dict[str, Dict[int, int]]] = None,
        val_loss_sum_per_cli: Optional[Dict[int, float]] = None,
        val_n_per_cli: Optional[Dict[int, int]] = None,
        signal_round: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        组装 LAB 完整诊断 dict (20 个核心字段 + 16 子字段).
        每个 dom 独立的字段会展开成 lab_<key>_<dom> 形式 (diagnostic.py 自动 dump scalar).

        Returns: dict, 含所有 LAB 诊断, append 到 model.proto_logs[-1].
        """
        diag = {}
        domains = lab_result["domains"]

        # ===== 核心 8 字段: 每个 domain 一个 scalar =====
        for d in domains:
            diag[f"lab_val_loss_ema_{d}"] = float(lab_result["loss_ema"][d])
            diag[f"lab_val_loss_raw_{d}"] = (
                float(self.val_loss_raw_history[d][-1])
                if self.val_loss_raw_history[d] else 0.0
            )
            diag[f"lab_gap_{d}"] = float(lab_result["gap"][d])
            diag[f"lab_q_{d}"] = float(lab_result["q"][d])
            diag[f"lab_w_proj_{d}"] = float(lab_result["w_proj"][d])
            diag[f"lab_ratio_{d}"] = float(lab_result["ratio"][d])
            diag[f"lab_boost_{d}"] = float(positive_delta.get(d, 0.0))
            diag[f"lab_cum_boost_{d}"] = float(sum(self.boost_history[d]))
            # delta_acc = acc[r] - acc[r-1]
            ah = self.acc_history[d]
            diag[f"lab_delta_acc_{d}"] = float(
                ah[-1] - ah[-2] if len(ah) >= 2 else 0.0
            )
            # clip status
            cs = lab_result["clip_status"][d]
            diag[f"lab_clip_at_max_{d}"] = 1.0 if cs == "@max" else 0.0
            diag[f"lab_clip_at_min_{d}"] = 1.0 if cs == "@min" else 0.0
            # 累计 clip count
            diag[f"lab_clip_max_count_{d}"] = float(self.clip_at_max_count[d])
            diag[f"lab_clip_min_count_{d}"] = float(self.clip_at_min_count[d])
            # sample share (固定不变, 但记录方便对比)
            diag[f"lab_sample_share_{d}"] = float(sample_share_dom[d])

        # ===== 全局元数据 =====
        diag["lab_round"] = int(round_idx)
        diag["lab_lambda"] = float(self.lam)
        diag["lab_ratio_min"] = float(self.ratio_min)
        diag["lab_ratio_max"] = float(self.ratio_max)
        diag["lab_ema_alpha"] = float(self.ema_alpha)
        diag["lab_n_iter"] = int(lab_result["n_iter"])
        diag["lab_converged"] = 1.0 if lab_result["converged"] else 0.0
        diag["lab_sum_check"] = float(lab_result["sum_check"])
        diag["lab_fallback_to_fedavg"] = 1.0 if lab_result.get("fallback_to_fedavg", False) else 0.0

        # signal source / round 元数据 (codex guardrail #2)
        diag["lab_signal_source"] = "client_val_ce_loss"   # 论文表述
        diag["lab_signal_round"] = int(signal_round if signal_round is not None else max(0, round_idx - 1))

        # ===== Val 元数据 (codex guardrail: val_class_counts) =====
        if val_n_per_dom is not None:
            for d, n in val_n_per_dom.items():
                diag[f"lab_val_n_{d}"] = int(n)
        if val_class_counts is not None:
            for d, cls_counts in val_class_counts.items():
                # 展开成 lab_val_class_count_{d}_c{cls}
                for c, cnt in cls_counts.items():
                    diag[f"lab_val_class_count_{d}_c{c}"] = int(cnt)
        if val_loss_sum_per_cli is not None:
            for k, v in val_loss_sum_per_cli.items():
                diag[f"lab_val_loss_sum_cli{k}"] = float(v)
        if val_n_per_cli is not None:
            for k, n in val_n_per_cli.items():
                diag[f"lab_val_n_cli{k}"] = int(n)

        # ===== 全局衍生指标 (会被 cold path 进一步分析) =====
        # 总 boost / 平均 boost / overall clip rate
        all_pd = [positive_delta.get(d, 0.0) for d in domains]
        diag["lab_total_boost_round"] = float(sum(all_pd))
        diag["lab_max_boost_round"] = float(max(all_pd) if all_pd else 0.0)
        # clip rate (本 round)
        n_clipped = sum(1 for d in domains if lab_result["clip_status"][d] is not None)
        diag["lab_clip_rate_round"] = float(n_clipped) / max(1, len(domains))

        return diag

    # ------------------------------------------------------------
    # ROI / waste 检测 (训练后或 stdout warning 用)
    # ------------------------------------------------------------
    def compute_window_roi(self, dom: str, round_idx: int) -> Optional[float]:
        """近 window_size round 的 ROI = Δacc_pp / mean_boost_pp."""
        ah = self.acc_history[dom]
        bh = self.boost_history[dom]
        if len(ah) < 2 or len(bh) < self.window_size:
            return None
        end = len(ah)
        start = max(0, end - self.window_size)
        delta_acc_pp = ah[end - 1] - ah[start]   # acc 已经是 0~100
        mean_boost = float(np.mean(bh[start:end]))
        if mean_boost < 1e-6:
            return None
        return delta_acc_pp / (mean_boost * 100.0)   # ratio 单位 pp/pp

    def compute_cumulative_roi(self, dom: str) -> Optional[float]:
        ah = self.acc_history[dom]
        bh = self.boost_history[dom]
        if len(ah) < 2:
            return None
        cum_boost = float(sum(bh))
        if cum_boost < 1e-6:
            return None
        acc_gain = ah[-1] - ah[0]
        return acc_gain / (cum_boost * 100.0)

    def detect_waste(self, round_idx: int, domains: List[str]) -> List[str]:
        """返回有 L2 浪费 (window_roi < threshold 持续 ≥ 2 个 window) 的 domain list."""
        wasted = []
        for d in domains:
            roi = self.compute_window_roi(d, round_idx)
            if roi is not None and roi < self.waste_roi_threshold:
                # 累计 cum_boost 至少 5% 才判定 (避免 cold start 误报)
                if sum(self.boost_history[d]) >= 0.05:
                    wasted.append(d)
        return wasted
