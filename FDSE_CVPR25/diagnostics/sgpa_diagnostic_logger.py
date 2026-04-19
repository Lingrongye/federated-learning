"""
FedDSA-SGPA 三层诊断监控器 (21 metrics).

设计理念: 不光记录 accuracy,而是记录从"训练梯度方向 → 特征几何 → 聚合漂移 → 推理 gate → 最终预测"
完整链路中间变量,让实验失败时能反推根因,不再是"测一个数就停"。

三层指标:
  Layer 1 (训练端):      6 个 — 特征几何 + 梯度健康度
  Layer 2 (聚合端):      2 个 — 跨 client 漂移度
  Layer 3 (推理/SGPA):   13 个 — gate 行为 + 原型质量 + 预测一致性

用法:
  dl = SGPADiagnosticLogger(client_id=0, stage='train', log_dir='./diag_logs')
  # 训练循环中
  metrics = {
      'orth':      dl.orthogonality(z_sem, z_sty),
      'etf_align': dl.etf_alignment(z_sem, labels, M, K)[0],
      ...
  }
  dl.record(round_id=r, metrics_dict=metrics)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SGPADiagnosticLogger:
    """三层诊断监控器。"""

    def __init__(self, client_id: int, stage: str, log_dir: str, dump_every_n: int = 5):
        assert stage in ('train', 'test', 'aggregate'), f"stage must be train/test/aggregate, got {stage}"
        self.client_id = client_id
        self.stage = stage
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dump_every = dump_every_n
        self.buf: List[dict] = []

    # =========================================================================
    # Layer 1: 训练端 6 个指标
    # =========================================================================

    @staticmethod
    def orthogonality(z_sem: torch.Tensor, z_sty: torch.Tensor) -> float:
        """Metric 1: 正交性 cos²(z_sem, z_sty).

        Args:
            z_sem, z_sty: [B, d] 双头输出.
        Returns:
            标量 float, 期望 [0.02, 0.15]. >0.3 → L_orth 被 CE 压制.
        """
        zs = F.normalize(z_sem, dim=-1)
        zt = F.normalize(z_sty, dim=-1)
        return (zs * zt).sum(-1).pow(2).mean().item()

    @staticmethod
    def etf_alignment(
        z_sem: torch.Tensor,
        labels: torch.Tensor,
        M: torch.Tensor,
        num_classes: int,
    ) -> Tuple[float, List[float]]:
        """Metric 2: ETF 对齐度 cos(z_sem_c_mean, M[:,c]) 每类 + 均值.

        Args:
            z_sem: [B, d], labels: [B], M: [d, K].
        Returns:
            (mean_align, per_class_list). 训练后期望 mean → 1.
        """
        d_feat, K = M.shape
        assert K == num_classes, f"M K={K} ≠ num_classes={num_classes}"
        per_class: List[float] = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            z_c_mean = z_sem[mask].mean(0)
            vertex = M[:, c]
            cos = F.cosine_similarity(
                z_c_mean.unsqueeze(0), vertex.unsqueeze(0)
            ).item()
            per_class.append(cos)
        mean_align = float(np.mean(per_class)) if per_class else 0.0
        return mean_align, per_class

    @staticmethod
    def intra_class_similarity(
        z_sem: torch.Tensor, labels: torch.Tensor, num_classes: int
    ) -> float:
        """Metric 3: 类内相似度 (pairwise cos 均值). 期望训练后 >0.6."""
        per_class: List[float] = []
        for c in range(num_classes):
            mask = labels == c
            n = int(mask.sum().item())
            if n < 2:
                continue
            z_c = F.normalize(z_sem[mask], dim=-1)
            sim = z_c @ z_c.T  # [n, n]
            off_diag_mean = (sim.sum() - sim.trace()) / (n * (n - 1))
            per_class.append(off_diag_mean.item())
        return float(np.mean(per_class)) if per_class else 0.0

    @staticmethod
    def inter_class_similarity(
        z_sem: torch.Tensor, labels: torch.Tensor, num_classes: int
    ) -> float:
        """Metric 4: 类间相似度 (centers pairwise cos 均值). 期望 <0.2."""
        centers = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            centers.append(F.normalize(z_sem[mask].mean(0), dim=-1))
        if len(centers) < 2:
            return 0.0
        C = torch.stack(centers)  # [K', d]
        sim = C @ C.T
        n = C.shape[0]
        return ((sim.sum() - sim.trace()) / (n * (n - 1))).item()

    @staticmethod
    def gradient_cosine(
        grad_ce_flat: torch.Tensor, grad_orth_flat: torch.Tensor
    ) -> float:
        """Metric 5: cos(∇L_CE, ∇L_orth). 期望 >0, <0 → 两 loss 打架."""
        return F.cosine_similarity(
            grad_ce_flat.unsqueeze(0), grad_orth_flat.unsqueeze(0)
        ).item()

    @staticmethod
    def gradient_norm_ratio(
        grad_ce_flat: torch.Tensor, grad_orth_flat: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """Metric 6: ‖∇L_CE‖ / ‖∇L_orth‖. 期望 [1, 10]."""
        return (grad_ce_flat.norm() / (grad_orth_flat.norm() + eps)).item()

    # =========================================================================
    # Layer 2: 聚合端 2 个指标
    # =========================================================================

    @staticmethod
    def client_center_variance(
        client_centers_list: List[torch.Tensor],
    ) -> float:
        """Metric 7: 跨 client 类中心方差. 期望训练中降低.

        Args:
            client_centers_list: list of [K, d] tensors, one per client.
        Returns:
            标量 float.
        """
        stacked = torch.stack(client_centers_list, dim=0)  # [N, K, d]
        var_across = stacked.var(dim=0, unbiased=False)  # [K, d]
        return var_across.mean().item()

    @staticmethod
    def param_drift(
        client_params: List[torch.Tensor], global_param: torch.Tensor
    ) -> float:
        """Metric 8: 客户端参数到全局参数的平均 L2 距离."""
        dists = [(p - global_param).norm().item() for p in client_params]
        return float(np.mean(dists))

    # =========================================================================
    # Layer 3: 推理/SGPA 13 个指标
    # =========================================================================

    @staticmethod
    def gate_rates(
        entropy: torch.Tensor, dist_min: torch.Tensor, tau_H: float, tau_S: float
    ) -> Dict[str, float]:
        """Metrics 9-11: reliable / entropy-only / dist-only gate 触发率.

        正常范围: reliable_rate ∈ [0.3, 0.7].
        """
        H_gate = entropy < tau_H
        D_gate = dist_min < tau_S
        reliable = H_gate & D_gate
        return {
            'reliable_rate': reliable.float().mean().item(),
            'entropy_rate': H_gate.float().mean().item(),
            'dist_rate': D_gate.float().mean().item(),
        }

    @staticmethod
    def dist_distribution(dist_min: torch.Tensor) -> Dict[str, float]:
        """Metric 12: dist_min 的分位数 {min, p10, p50, p90, max}."""
        arr = dist_min.detach().cpu().numpy()
        return {
            'dist_min_min': float(arr.min()),
            'dist_min_p10': float(np.percentile(arr, 10)),
            'dist_min_p50': float(np.percentile(arr, 50)),
            'dist_min_p90': float(np.percentile(arr, 90)),
            'dist_min_max': float(arr.max()),
        }

    @staticmethod
    def whitening_reduction(
        z_sty_raw: torch.Tensor,
        z_sty_white: torch.Tensor,
        source_mu_raw: torch.Tensor,
        source_mu_white: torch.Tensor,
    ) -> Dict[str, float]:
        """Metric 13: 白化前后跨 client dist 散度对比.

        每个 test 样本对 N 个 source μ 的 dist 数组,算 std.
        白化后该 std 应显著降低.
        """
        d_raw = ((z_sty_raw.unsqueeze(1) - source_mu_raw.unsqueeze(0)) ** 2).sum(-1)
        d_white = ((z_sty_white.unsqueeze(1) - source_mu_white.unsqueeze(0)) ** 2).sum(-1)
        raw_std = d_raw.std(-1, unbiased=False).mean().item()
        white_std = d_white.std(-1, unbiased=False).mean().item()
        return {
            'raw_scatter': raw_std,
            'white_scatter': white_std,
            'reduction_ratio': white_std / (raw_std + 1e-8),
        }

    @staticmethod
    def sigma_condition_number(Sigma: torch.Tensor) -> float:
        """Metric 14: cond(Σ_global). >1e6 → 病态.

        使用 SVD-based cond (稳定), 避免 torch.linalg.cond 在 CPU/GPU 差异.
        """
        # 强制转 float 避免 double precision 问题
        Sigma_f = Sigma.float()
        s = torch.linalg.svdvals(Sigma_f)
        return (s.max() / (s.min() + 1e-12)).item()

    @staticmethod
    def proto_fill(
        supports_dict: Dict[int, List], num_classes: int
    ) -> Tuple[Dict[int, int], float]:
        """Metric 15: 每类 proto 支撑样本数 + 均值.

        Args:
            supports_dict: {class_id: list of supports}.
        Returns:
            (per_class_fill_dict, mean_fill).
        """
        fill = {c: len(supports_dict.get(c, [])) for c in range(num_classes)}
        mean_fill = float(np.mean(list(fill.values()))) if fill else 0.0
        return fill, mean_fill

    @staticmethod
    def proto_etf_offset(
        proto: List[Optional[torch.Tensor]], M: torch.Tensor, num_classes: int
    ) -> Tuple[float, List[float]]:
        """Metric 16: 1 - cos(proto[c], M[:,c]) 每类 + 均值.

        0 = proto 与 ETF 顶点完全对齐 (没校正).
        ~1 = 正交 (校正大).
        2 = 反向 (可能 bug).
        """
        offsets: List[float] = []
        for c in range(num_classes):
            p = proto[c] if c < len(proto) else None
            if p is None or p.numel() == 0 or p.norm() < 1e-6:
                offsets.append(0.0)
                continue
            cos = F.cosine_similarity(
                p.unsqueeze(0), M[:, c].unsqueeze(0)
            ).item()
            offsets.append(1.0 - cos)
        return float(np.mean(offsets)) if offsets else 0.0, offsets

    @staticmethod
    def fallback_rate(activated: torch.Tensor) -> float:
        """Metric 17: 使用 ETF fallback 的样本比例.

        高 fallback rate 说明 SGPA 还没接管.
        """
        return (1.0 - activated.float()).mean().item()

    @staticmethod
    def prediction_accuracy(
        pred_proto: torch.Tensor,
        pred_etf: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Metrics 18-21: proto_acc, etf_acc, agreement, proto_vs_etf_gain.

        如果 labels 为 None,只返回 agreement.
        """
        agree = (pred_proto == pred_etf).float().mean().item()
        result: Dict[str, float] = {'pred_agree': agree}
        if labels is not None:
            proto_acc = (pred_proto == labels).float().mean().item()
            etf_acc = (pred_etf == labels).float().mean().item()
            result.update({
                'proto_acc': proto_acc,
                'etf_acc': etf_acc,
                'proto_vs_etf_gain': proto_acc - etf_acc,
            })
        return result

    # =========================================================================
    # 接口: record + dump
    # =========================================================================

    def record(self, round_id: int, metrics_dict: dict) -> None:
        """把一次测量存进 buffer, 到 dump_every_n 次就 flush."""
        entry = dict(metrics_dict)  # copy, 避免外部修改
        entry['_round'] = round_id
        entry['_client'] = self.client_id
        entry['_stage'] = self.stage
        self.buf.append(entry)
        if len(self.buf) >= self.dump_every:
            self.dump()

    def dump(self) -> None:
        """Append buffer 到 jsonl file, 清空 buffer."""
        if not self.buf:
            return
        path = self.log_dir / f"diag_{self.stage}_client{self.client_id}.jsonl"
        with open(path, 'a', encoding='utf-8') as f:
            for entry in self.buf:
                f.write(json.dumps(entry, default=_json_default, ensure_ascii=False) + '\n')
        self.buf.clear()


def _json_default(o):
    """Serialize torch tensor / numpy as scalar."""
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.item()
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
