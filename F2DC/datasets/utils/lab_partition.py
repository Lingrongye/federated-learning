"""
LAB v4.2 Validation Partition — 给已构造好的 trainloaders 加 client-held val
=================================================================================

设计原则 (PROPOSAL §3.2):
1. **Val 来源**: 仅来自每个 cli 训练 dataset 的"未被任何 client 选中"的 unused 部分.
   对 PACS/Office: 这是 ImageFolder_Custom.train_index_list 内的 unused.
   对 Digits MNIST/USPS/SVHN: 这是 raw torchvision dataset 的 unused index.
   对 Digits syn: 这是 ImageFolder 的 unused index.
   绝不碰 test split.
2. **反推 unused (codex guardrail #1)**:
   used = union(client_selected_idx for clients in this domain)
   unused = arange(len(inner_dataset)) - used
   不直接信任 partition 函数返回的 not_used_index_dict (含 class repair).
3. **Stratified per-class**: 每类最多 5 张, 每域 cap = min(val_size_per_dom, C × per_class).
   实际单域 val 大小 (PROPOSAL §3.3 修正后):
     - PACS  (C=7):  cap = min(50, 7*5)  = 35 (受 stratified 限制)
     - Office (C=10): cap = min(50, 10*5) = 50
     - Digits (C=10): cap = min(50, 10*5) = 50
   注: codex 三轮 Important #2 指出 PACS val 实际 35 而非 50,
       这是 stratified per-class=5 的硬约束, 不破坏均衡性.
4. **Deterministic eval transform**: 不能用 train 的 random crop/flip.
5. **Shard 给该域所有 cli**: cli k 持 val_pool[d][shard_idx], 加权聚合.
6. **Val seed 固定 42** (跟 train seed 解耦, 跨实验 val 集一致).
7. **跨 dataset 兼容** (codex Critical #3): MNIST/USPS/SVHN/ImageFolder/ImageFolder_Custom
   都通过 _DeterministicValWrapper 走通用路径, 共享 inner dataset 的 raw 数据,
   override transform 走 deterministic eval.

接口:
    val_loaders, val_meta = setup_lab_val_loaders(
        trainloaders, train_dataset_list, selected_domain_list, args
    )
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 默认配置 (跟 PROPOSAL §10 一致)
DEFAULT_VAL_SIZE_PER_DOM = 50
DEFAULT_VAL_PER_CLASS = 5
DEFAULT_VAL_SEED = 42


# =============================================================================
# Eval transform builder
# =============================================================================
def _build_eval_transform(dataset_name: str) -> transforms.Compose:
    """构造 deterministic eval transform (跟 dataset 现有 test_transform 一致, 无 random aug)."""
    if dataset_name == "fl_pacs":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif dataset_name == "fl_officecaltech":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif dataset_name == "fl_digits":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        raise ValueError(f"Unknown dataset {dataset_name} for LAB val transform")


# =============================================================================
# Universal val Subset wrapper
# =============================================================================
class _DeterministicValWrapper(Dataset):
    """
    共享 inner dataset 的 raw data, 但 override transform 走 deterministic eval.

    支持的 inner dataset 类型 (按属性探测, 顺序敏感):
      1. ImageFolder_Custom (PACS/Office):
         有 .imagefolder_obj + .train_index_list
         indices 是 *相对* train_index_list 的 rel index (跟 sampler.indices 一致)
         转绝对: abs = train_index_list[rel], 再用 imagefolder_obj.samples[abs]
      2. ImageFolder (Digits syn):
         有 .samples + .loader. indices 是绝对 index.
      3. SVHN:
         有 .data ((N,3,H,W) uint8) + .labels (numpy). indices 是绝对 index.
      4. MNIST/USPS:
         有 .data ((N,H,W) tensor uint8) + .targets (list/tensor). indices 是绝对 index.

    完全不复制 raw data (内存零开销), 只持引用 + indices + eval_transform.
    """

    def __init__(self, inner_dataset, indices, eval_transform):
        self.inner = inner_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.eval_transform = eval_transform
        self._extractor = self._detect_extractor(inner_dataset)

    @staticmethod
    def _detect_extractor(ds) -> str:
        # 顺序敏感: ImageFolder_Custom 必须先于 ImageFolder, 因为它内含 imagefolder_obj
        if hasattr(ds, "imagefolder_obj") and hasattr(ds, "train_index_list"):
            return "imagefolder_custom"
        if hasattr(ds, "samples") and hasattr(ds, "loader"):
            return "imagefolder"
        if hasattr(ds, "data") and hasattr(ds, "labels"):
            return "svhn"
        if hasattr(ds, "data") and hasattr(ds, "targets"):
            return "mnist_usps"
        return "unknown"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        if self._extractor == "imagefolder_custom":
            # PACS/Office: indices 是相对 train_index_list 的 rel index, 转绝对
            abs_idx = int(self.inner.train_index_list[idx])
            path, target = self.inner.imagefolder_obj.samples[abs_idx]
            img = self.inner.imagefolder_obj.loader(path)
        elif self._extractor == "imagefolder":
            path, target = self.inner.samples[idx]
            img = self.inner.loader(path)
        elif self._extractor == "mnist_usps":
            data = self.inner.data[idx]
            target = self.inner.targets[idx]
            if hasattr(data, "numpy"):
                data = data.numpy()
            mode = "L" if data.ndim == 2 else "RGB"
            img = Image.fromarray(data, mode=mode)
        elif self._extractor == "svhn":
            data = self.inner.data[idx]
            target = self.inner.labels[idx]
            if data.ndim == 3 and data.shape[0] == 3:
                data = data.transpose(1, 2, 0)
            img = Image.fromarray(data, mode="RGB")
        else:
            raise RuntimeError(
                f"LAB val: unknown inner dataset type {type(self.inner).__name__}, "
                f"missing 'imagefolder_obj+train_index_list' / 'samples+loader' / "
                f"'data+targets' / 'data+labels' attrs"
            )

        target = int(target) if hasattr(target, "__int__") else int(target.item() if hasattr(target, "item") else target)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.eval_transform is not None:
            img = self.eval_transform(img)
        return img, target


# =============================================================================
# 提取 dataset 的 size / targets (用于 stratified)
# =============================================================================
def _get_dataset_size(inner_dataset) -> int:
    """通用拿 dataset N (相对 sampler.indices 的范围)."""
    # ImageFolder_Custom: sampler.indices 在 train_index_list 内, N = len(train_index_list)
    if hasattr(inner_dataset, "imagefolder_obj") and hasattr(inner_dataset, "train_index_list"):
        return int(len(inner_dataset.train_index_list))
    return int(len(inner_dataset))


def _extract_targets(inner_dataset) -> np.ndarray:
    """通用提取 targets array, 顺序对应 sampler.indices 的相对索引."""
    # 顺序敏感: ImageFolder_Custom 优先
    if hasattr(inner_dataset, "imagefolder_obj") and hasattr(inner_dataset, "train_index_list"):
        all_targets = inner_dataset.imagefolder_obj.targets
        if hasattr(all_targets, "numpy"):
            all_targets = all_targets.numpy()
        all_targets = np.asarray(all_targets, dtype=np.int64)
        # train_index_list 是绝对 index 进 imagefolder; 我们要按 rel idx 顺序
        train_idx = np.asarray(inner_dataset.train_index_list, dtype=np.int64)
        return all_targets[train_idx]   # shape (len(train_index_list),) 按 rel idx 顺序
    if hasattr(inner_dataset, "labels"):
        # SVHN: numpy
        labels = inner_dataset.labels
        return np.asarray(labels, dtype=np.int64)
    if hasattr(inner_dataset, "targets"):
        # MNIST/USPS/ImageFolder: list or tensor
        targets = inner_dataset.targets
        if hasattr(targets, "numpy"):
            targets = targets.numpy()
        return np.asarray(targets, dtype=np.int64)
    if hasattr(inner_dataset, "samples"):
        # ImageFolder fallback: samples = list of (path, target)
        return np.asarray([s[1] for s in inner_dataset.samples], dtype=np.int64)
    raise RuntimeError(f"LAB val: cannot extract targets from {type(inner_dataset).__name__}")


# =============================================================================
# Stratified sampling
# =============================================================================
def _stratified_sample_indices(
    candidate_idx: np.ndarray,
    targets_for_candidates: np.ndarray,
    per_class: int,
    max_total: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    从 candidate_idx 里 stratified 抽 val:
      - 每类最多 per_class 张
      - 总数最多 max_total 张
      - rng 是 deterministic (seed=42)
    """
    classes = np.unique(targets_for_candidates)
    selected = []
    class_counts: Dict[int, int] = {}
    for c in sorted(classes.tolist()):
        mask = (targets_for_candidates == c)
        idx_c = candidate_idx[mask]
        if len(idx_c) == 0:
            class_counts[c] = 0
            continue
        n_pick = min(int(per_class), len(idx_c))
        perm = rng.permutation(len(idx_c))[:n_pick]
        picked = idx_c[perm]
        selected.append(picked)
        class_counts[c] = n_pick
    if not selected:
        return np.array([], dtype=np.int64), class_counts

    selected_idx = np.concatenate(selected)
    if len(selected_idx) > max_total:
        perm = rng.permutation(len(selected_idx))[:max_total]
        selected_idx = selected_idx[perm]
        # 重新算 class_counts
        cand_to_target = dict(zip(candidate_idx.tolist(), targets_for_candidates.tolist()))
        new_counts: Dict[int, int] = defaultdict(int)
        for ai in selected_idx.tolist():
            new_counts[int(cand_to_target[ai])] += 1
        class_counts = dict(new_counts)
    return selected_idx, class_counts


# =============================================================================
# Main entry: setup val loaders
# =============================================================================
def setup_lab_val_loaders(
    trainloaders: List[DataLoader],
    train_dataset_list: List[Any],
    selected_domain_list: List[str],
    args,
    val_size_per_dom: int = DEFAULT_VAL_SIZE_PER_DOM,
    val_per_class: int = DEFAULT_VAL_PER_CLASS,
    val_seed: int = DEFAULT_VAL_SEED,
) -> Tuple[List[Optional[DataLoader]], Dict[str, Any]]:
    """
    主入口: 给已构造好的 trainloaders 加 LAB val_loaders.

    Args:
        trainloaders: list[parti_num] of DataLoader
        train_dataset_list: list[parti_num] of inner dataset
            (PACS/Office: ImageFolder_Custom; Digits: raw MNIST/USPS/SVHN or ImageFolder)
        selected_domain_list: cli i 持有的 domain name (按 cli_id 顺序)
        args: 主参数 (用于读 dataset name + parti_num + local_batch_size)

    Returns:
        val_loaders: list[parti_num] of DataLoader (cli 没分到 val 时为 None)
        val_meta: dict {
            'val_n_per_cli', 'val_n_per_dom', 'val_class_counts',
            'val_seed', 'val_size_per_dom', 'val_per_class',
            'val_pool_origin', 'val_transform',
        }
    """
    parti_num = int(args.parti_num)
    dataset_name = str(args.dataset)
    rng = np.random.RandomState(val_seed)

    # ===== Step A: 收集每 cli 的 selected indices (从 trainloader.sampler.indices) =====
    cli_selected_idx: Dict[int, np.ndarray] = {}
    for k, dl in enumerate(trainloaders):
        if dl is None or not hasattr(dl, "sampler"):
            cli_selected_idx[k] = np.array([], dtype=np.int64)
            continue
        sampler = dl.sampler
        if hasattr(sampler, "indices"):
            cli_selected_idx[k] = np.asarray(sampler.indices, dtype=np.int64)
        else:
            cli_selected_idx[k] = np.array([], dtype=np.int64)

    # ===== Step B: 按 domain 组内反推 unused =====
    domain_to_cli: Dict[str, List[int]] = defaultdict(list)
    for k, d in enumerate(selected_domain_list):
        domain_to_cli[str(d)].append(int(k))

    val_loaders: List[Optional[DataLoader]] = [None] * parti_num
    val_n_per_cli: Dict[int, int] = {}
    val_n_per_dom: Dict[str, int] = {}
    val_class_counts: Dict[str, Dict[int, int]] = {}

    eval_transform = _build_eval_transform(dataset_name)

    for dom_name, cli_list in domain_to_cli.items():
        if not cli_list:
            continue
        # 拿 domain 的 inner dataset (从该域第一个 cli 的 trainloader.dataset)
        first_cli = cli_list[0]
        inner_dataset = train_dataset_list[first_cli]

        # Step C: 算 used (该 domain 内所有 cli 的 selected_idx 并集)
        used_set = set()
        for k in cli_list:
            used_set.update(cli_selected_idx[k].tolist())

        # Step D: 算 unused (相对 sampler.indices 的范围)
        # PACS/Office (ImageFolder_Custom): sampler.indices 是 train_index_list 内的 rel idx,
        #   所以 N = len(train_index_list), unused 也是 rel idx.
        # Digits (raw torchvision/ImageFolder): sampler.indices 是绝对 idx, N = len(inner).
        try:
            n_inner = _get_dataset_size(inner_dataset)
        except Exception as e:
            print(f"[LAB partition WARN] domain {dom_name} can't get dataset size: {e}")
            continue
        all_rel = np.arange(n_inner, dtype=np.int64)
        unused = np.array([i for i in all_rel if int(i) not in used_set], dtype=np.int64)

        if len(unused) == 0:
            print(f"[LAB partition WARN] domain {dom_name} has no unused index, val skipped.")
            continue

        # Step E: 拿到 inner dataset 全部样本的 targets (按 rel idx 顺序, 跟 unused 配对)
        try:
            all_targets = _extract_targets(inner_dataset)
        except Exception as e:
            print(f"[LAB partition WARN] domain {dom_name} can't extract targets: {e}, val skipped.")
            continue
        if len(all_targets) != n_inner:
            print(f"[LAB partition WARN] domain {dom_name} targets len mismatch "
                  f"(targets={len(all_targets)}, dataset={n_inner}), val skipped.")
            continue
        unused_targets = all_targets[unused]

        # Step F: stratified 抽 val
        cap_total = min(int(val_size_per_dom), len(unused))
        sel_idx, cls_counts = _stratified_sample_indices(
            candidate_idx=unused,
            targets_for_candidates=unused_targets,
            per_class=val_per_class,
            max_total=cap_total,
            rng=rng,
        )
        if len(sel_idx) == 0:
            print(f"[LAB partition WARN] domain {dom_name} stratified got 0, val skipped.")
            continue

        # Step G: shard 给该域所有 cli
        n_cli = len(cli_list)
        sel_idx_perm = sel_idx.copy()
        rng.shuffle(sel_idx_perm)
        shards = np.array_split(sel_idx_perm, n_cli)

        val_n_per_dom[dom_name] = int(len(sel_idx))
        val_class_counts[dom_name] = {int(c): int(cnt) for c, cnt in cls_counts.items()}

        for shard_idx, k in enumerate(cli_list):
            shard_indices = shards[shard_idx].astype(np.int64)
            if len(shard_indices) == 0:
                val_loaders[k] = None
                val_n_per_cli[k] = 0
                continue
            val_ds = _DeterministicValWrapper(
                inner_dataset=inner_dataset,
                indices=shard_indices,
                eval_transform=eval_transform,
            )
            batch_size = int(getattr(args, "local_batch_size", 64))
            val_loaders[k] = DataLoader(
                val_ds,
                batch_size=min(batch_size, len(shard_indices)),
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            val_n_per_cli[k] = int(len(shard_indices))

    val_meta = {
        "val_n_per_cli": val_n_per_cli,
        "val_n_per_dom": val_n_per_dom,
        "val_class_counts": val_class_counts,
        "val_seed": int(val_seed),
        "val_size_per_dom": int(val_size_per_dom),
        "val_per_class": int(val_per_class),
        "val_pool_origin": "train_index_unused_via_used_complement",
        "val_transform": "deterministic_eval",
    }
    return val_loaders, val_meta


# =============================================================================
# Forward eval helper (PG-DFC backbone is_eval=True 走 deterministic gumbel)
# =============================================================================
def _forward_eval(net, images):
    """
    Forward backbone with deterministic eval path (无 gumbel 随机).
    PG-DFC backbone forward signature: net(images, is_eval=True) → 7-tuple.
    其它 backbone: net(images) → tensor 或 tuple.

    返回 logits (tensor of shape (B, num_classes)).
    """
    try:
        forward_code = net.forward.__code__
        if 'is_eval' in forward_code.co_varnames:
            output = net(images, is_eval=True)
        else:
            output = net(images)
    except (AttributeError, TypeError):
        output = net(images)

    # PG-DFC backbone 返回 7-tuple (out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten)
    if isinstance(output, tuple):
        return output[0]
    return output


# =============================================================================
# Eval val per client
# =============================================================================
def evaluate_val_per_client(
    global_net: torch.nn.Module,
    val_loaders: List[Optional[DataLoader]],
    device,
    selected_domain_list: List[str],
) -> Dict[str, Any]:
    """
    用 global_net 在每个 cli 的 val_loader 上跑 CE loss, 返回:
        {
            'val_loss_per_dom':  {dom: sample_weighted_avg_loss},
            'val_loss_sum_per_cli': {cli_id: sum_of_per_sample_loss},
            'val_n_per_cli': {cli_id: n_samples},
        }
    用 sample-weighted aggregation (codex guardrail).
    用 deterministic eval forward (is_eval=True) 避免 gumbel 噪声 (codex Important #4).
    """
    was_training = global_net.training
    global_net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

    val_loss_sum_per_cli: Dict[int, float] = {}
    val_n_per_cli: Dict[int, int] = {}

    try:
        with torch.no_grad():
            for k, vl in enumerate(val_loaders):
                if vl is None:
                    continue
                sum_loss = 0.0
                n_total = 0
                for images, labels in vl:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = _forward_eval(global_net, images)
                    loss = criterion(logits, labels)
                    sum_loss += float(loss.item())
                    n_total += int(labels.size(0))
                if n_total > 0:
                    val_loss_sum_per_cli[k] = sum_loss
                    val_n_per_cli[k] = n_total
    finally:
        if was_training:
            global_net.train()

    dom_to_cli: Dict[str, List[int]] = defaultdict(list)
    for k, d in enumerate(selected_domain_list):
        if k in val_n_per_cli:
            dom_to_cli[str(d)].append(int(k))

    val_loss_per_dom: Dict[str, float] = {}
    for d, cli_list in dom_to_cli.items():
        total_loss = sum(val_loss_sum_per_cli[k] for k in cli_list)
        total_n = sum(val_n_per_cli[k] for k in cli_list)
        if total_n > 0:
            val_loss_per_dom[d] = total_loss / total_n
        else:
            val_loss_per_dom[d] = 0.0

    return {
        "val_loss_per_dom": val_loss_per_dom,
        "val_loss_sum_per_cli": val_loss_sum_per_cli,
        "val_n_per_cli": val_n_per_cli,
    }
