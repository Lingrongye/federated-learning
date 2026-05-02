"""
Sanity tests for LAB v4.2 implementation
==========================================

不依赖 GPU / 真实 dataset, 纯 Python 单元测试.

覆盖:
  T1: bounded_simplex_projection 数值正确性 (5 个 corner case)
  T2: lab_step gap=0 严格退化 FedAvg (codex blocking fix)
  T3: lab_step PACS realistic (跟 P0 expected 一致)
  T4: LabState EMA 行为正确
  T5: LabState ROI / waste 检测
  T6: lab_partition stratified sampling 不重叠 train_idx (codex guardrail #1)
  T7: lab_partition deterministic eval transform (无 RandomCrop/RandomHorizontalFlip)
  T8: F2DCPgLab class 可 import + NAME 正确

用法:
  cd F2DC
  python test_lab_sanity.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ===== T1-T3: lab_aggregation =====
from models.utils.lab_aggregation import (
    bounded_simplex_projection, lab_step, LabState
)


def t1_projection_corners():
    """T1: bounded_simplex_projection 5 个 corner case."""
    print("\n=== T1: bounded_simplex_projection corners ===")
    passed = 0; total = 0

    # T1a: degenerate FedAvg (w_raw 已经 sum=1, 在 bounds 内)
    w_raw = np.array([0.20, 0.30, 0.25, 0.25])
    shares = w_raw.copy()
    w_min = 0.80 * shares; w_max = 2.00 * shares
    w_proj, n_iter, ok = bounded_simplex_projection(w_raw, w_min, w_max)
    total += 1
    if abs(w_proj.sum() - 1.0) < 1e-8 and np.allclose(w_proj, w_raw, atol=1e-8):
        passed += 1; print(f"  ✅ T1a degenerate: sum={w_proj.sum():.8f}, n_iter={n_iter}")
    else:
        print(f"  ❌ T1a degenerate: w_proj={w_proj}, sum={w_proj.sum()}")

    # T1b: small dom hits ratio_max
    shares = np.array([0.10, 0.20, 0.20, 0.50])
    w_raw = (1-0.5) * shares + 0.5 * np.array([1.0, 0, 0, 0])
    w_min = 0.80 * shares; w_max = 2.00 * shares
    w_proj, _, _ = bounded_simplex_projection(w_raw, w_min, w_max)
    total += 1
    if abs(w_proj[0] - w_max[0]) < 1e-5 and abs(w_proj.sum() - 1.0) < 1e-8:
        passed += 1; print(f"  ✅ T1b small_dom_max: w_proj[0]={w_proj[0]:.4f} hits w_max={w_max[0]:.4f}")
    else:
        print(f"  ❌ T1b small_dom_max: w_proj[0]={w_proj[0]:.4f}")

    # T1c: large dom hits ratio_min
    shares = np.array([0.10, 0.10, 0.10, 0.70])
    w_raw = (1-0.5) * shares + 0.5 * np.array([0.5, 0.5, 0, 0])
    w_min = 0.80 * shares; w_max = 2.00 * shares
    w_proj, _, _ = bounded_simplex_projection(w_raw, w_min, w_max)
    total += 1
    if abs(w_proj[3] - w_min[3]) < 1e-3 and abs(w_proj.sum() - 1.0) < 1e-8:
        passed += 1; print(f"  ✅ T1c large_dom_min: w_proj[3]={w_proj[3]:.4f} hits w_min={w_min[3]:.4f}")
    else:
        print(f"  ❌ T1c large_dom_min: w_proj[3]={w_proj[3]:.4f}")

    # T1d: extreme w_raw (negative values, codex bounds 鲁棒性测试)
    w_raw = np.array([-0.5, -0.5, -0.5, 1.5])
    shares = np.array([0.1, 0.2, 0.3, 0.4])
    w_min = 0.80 * shares; w_max = 2.00 * shares
    w_proj, n_iter, ok = bounded_simplex_projection(w_raw, w_min, w_max)
    total += 1
    in_bounds = (w_proj >= w_min - 1e-6).all() and (w_proj <= w_max + 1e-6).all()
    if in_bounds and abs(w_proj.sum() - 1.0) < 1e-6:
        passed += 1; print(f"  ✅ T1d extreme: sum={w_proj.sum():.8f}, n_iter={n_iter}, all in bounds")
    else:
        print(f"  ❌ T1d extreme: w_proj={w_proj}, in_bounds={in_bounds}")

    # T1e: random fuzz
    rng = np.random.RandomState(0)
    fuzz_pass = 0
    for _ in range(20):
        s = rng.dirichlet(np.ones(4))
        w_raw = (1-0.15) * s + 0.15 * rng.dirichlet(np.ones(4))
        w_min = 0.80 * s; w_max = 2.00 * s
        w_proj, _, _ = bounded_simplex_projection(w_raw, w_min, w_max)
        if (abs(w_proj.sum() - 1.0) < 1e-6
                and (w_proj >= w_min - 1e-6).all()
                and (w_proj <= w_max + 1e-6).all()):
            fuzz_pass += 1
    total += 1
    if fuzz_pass == 20:
        passed += 1; print(f"  ✅ T1e fuzz: 20/20 random Dirichlet runs passed")
    else:
        print(f"  ❌ T1e fuzz: {fuzz_pass}/20")

    return passed, total


def t2_gap_zero_fedavg():
    """T2: gap=0 严格退化 FedAvg (codex blocking fix)."""
    print("\n=== T2: lab_step gap=0 strict FedAvg degeneration ===")
    passed = 0; total = 0

    # T2a: 4 个 domain loss 完全相等
    sample_share = {"a": 0.128, "b": 0.237, "c": 0.180, "d": 0.455}
    # sum=1.0
    loss_uniform = {d: 0.30 for d in sample_share}
    result = lab_step(loss_uniform, sample_share)
    ratios = np.array([result["ratio"][d] for d in sample_share])
    w_proj = np.array([result["w_proj"][d] for d in sample_share])
    shares_arr = np.array([sample_share[d] for d in sample_share])
    total += 1
    if (np.allclose(ratios, 1.0, atol=1e-9)
            and np.allclose(w_proj, shares_arr, atol=1e-9)):
        passed += 1
        print(f"  ✅ T2a uniform_loss: ratio range [{ratios.min():.10f}, {ratios.max():.10f}]")
    else:
        print(f"  ❌ T2a uniform_loss: ratio range [{ratios.min():.6f}, {ratios.max():.6f}]")

    # T2b: 所有 loss 都 < mean (gap 全 0)
    loss_low = {"a": 0.10, "b": 0.10, "c": 0.10, "d": 0.10}
    r2 = lab_step(loss_low, sample_share)
    ratios2 = np.array([r2["ratio"][d] for d in sample_share])
    total += 1
    if np.allclose(ratios2, 1.0, atol=1e-9):
        passed += 1; print(f"  ✅ T2b all_low_loss: ratio all 1.0")
    else:
        print(f"  ❌ T2b all_low_loss: ratios={ratios2}")

    return passed, total


def t3_pacs_realistic():
    """T3: PACS realistic 公式跟 P0 expected 一致."""
    print("\n=== T3: PACS realistic (跟 P0 一致) ===")
    passed = 0; total = 0
    sample_share = {"photo": 0.1284, "art": 0.2370, "cartoon": 0.1804, "sketch": 0.4542}
    val_loss = {"photo": 0.325, "art": 0.418, "cartoon": 0.215, "sketch": 0.162}
    expected = {"photo": 1.14, "art": 1.32, "cartoon": 0.85, "sketch": 0.85}
    result = lab_step(val_loss, sample_share)
    for d, exp in expected.items():
        actual = result["ratio"][d]
        total += 1
        if abs(actual - exp) < 0.05:
            passed += 1; print(f"  ✅ {d}: ratio={actual:.3f} (expected ≈{exp:.2f})")
        else:
            print(f"  ❌ {d}: ratio={actual:.3f} (expected ≈{exp:.2f})")
    return passed, total


def t4_labstate_ema():
    """T4: LabState EMA 行为正确."""
    print("\n=== T4: LabState EMA correctness ===")
    state = LabState(ema_alpha=0.3)
    state.update_val_loss(1, {"photo": 1.0, "art": 1.0, "cartoon": 1.0, "sketch": 1.0})
    state.update_val_loss(2, {"photo": 0.0, "art": 0.0, "cartoon": 0.0, "sketch": 0.0})
    # ema after 2 updates: 0.3*0 + 0.7*1.0 = 0.7
    expected_ema = 0.7
    passed = 0; total = 1
    if abs(state.val_loss_ema["photo"] - expected_ema) < 1e-9:
        passed += 1; print(f"  ✅ EMA after 2 updates: {state.val_loss_ema['photo']:.4f} (expected {expected_ema})")
    else:
        print(f"  ❌ EMA after 2 updates: {state.val_loss_ema['photo']:.4f}")
    return passed, total


def t5_labstate_compute_lab():
    """T5: LabState.compute_lab 第一轮 fallback FedAvg, 第二轮起用 LAB."""
    print("\n=== T5: LabState round 1 fallback ===")
    state = LabState()
    sample_share = {"a": 0.5, "b": 0.5}
    # Round 1: 还没有 val_loss
    r1 = state.compute_lab(round_idx=1, sample_share_dom=sample_share)
    passed = 0; total = 0
    total += 1
    if r1.get("fallback_to_fedavg", False) and abs(r1["ratio"]["a"] - 1.0) < 1e-9:
        passed += 1; print(f"  ✅ Round 1 fallback: ratio=1.0, fallback={r1['fallback_to_fedavg']}")
    else:
        print(f"  ❌ Round 1 fallback failed")
    # Round 2: 有 val_loss
    state.update_val_loss(1, {"a": 0.3, "b": 0.5})
    r2 = state.compute_lab(round_idx=2, sample_share_dom=sample_share)
    total += 1
    if not r2["fallback_to_fedavg"] and r2["ratio"]["a"] < 1.0 and r2["ratio"]["b"] > 1.0:
        passed += 1; print(f"  ✅ Round 2 LAB: ratio={{a:{r2['ratio']['a']:.3f}, b:{r2['ratio']['b']:.3f}}}")
    else:
        print(f"  ❌ Round 2 LAB direction wrong")
    return passed, total


def t6_lab_partition_stratify():
    """T6: lab_partition stratified sampling, val 跟 train idx 不重叠."""
    print("\n=== T6: lab_partition stratified sampling ===")
    from datasets.utils.lab_partition import _stratified_sample_indices

    rng = np.random.RandomState(42)
    # 模拟 64 张样本, 10 类
    candidate_abs = np.arange(64)
    targets = np.array([i % 10 for i in range(64)])  # 0,1,...,9,0,1,... (每类 6-7 张)
    selected, class_counts = _stratified_sample_indices(
        candidate_idx=candidate_abs,
        targets_for_candidates=targets,
        per_class=5,
        max_total=50,
        rng=rng,
    )
    passed = 0; total = 0
    total += 1
    if len(selected) <= 50 and all(c <= 5 for c in class_counts.values()):
        passed += 1; print(f"  ✅ stratified: {len(selected)} total, class_counts={dict(sorted(class_counts.items()))}")
    else:
        print(f"  ❌ stratified violated: {len(selected)}, {class_counts}")

    # T6b: 极端小池 (Office dslr 模拟): 64 candidates, 不可能每类 5 张
    candidate_small = np.arange(20)   # 20 张
    targets_small = np.array([i % 10 for i in range(20)])  # 每类 2 张
    sel2, cc2 = _stratified_sample_indices(
        candidate_idx=candidate_small,
        targets_for_candidates=targets_small,
        per_class=5,
        max_total=50,
        rng=rng,
    )
    total += 1
    if len(sel2) == 20 and all(c == 2 for c in cc2.values()):
        passed += 1; print(f"  ✅ small_pool: got all {len(sel2)} samples (per_class capped at 2)")
    else:
        print(f"  ❌ small_pool: {len(sel2)}, {cc2}")
    return passed, total


def t7_eval_transform_deterministic():
    """T7: eval_transform 不含 RandomCrop / RandomHorizontalFlip."""
    print("\n=== T7: deterministic eval transform ===")
    from datasets.utils.lab_partition import _build_eval_transform

    passed = 0; total = 0
    for ds_name in ["fl_pacs", "fl_officecaltech", "fl_digits"]:
        et = _build_eval_transform(ds_name)
        ts_str = str(et)
        total += 1
        has_random = "RandomCrop" in ts_str or "RandomHorizontalFlip" in ts_str
        if not has_random:
            passed += 1; print(f"  ✅ {ds_name}: deterministic ({ts_str.replace(chr(10), ' ').strip()[:80]}...)")
        else:
            print(f"  ❌ {ds_name}: contains random aug ({ts_str})")
    return passed, total


def t9_wrapper_compat():
    """T9: _DeterministicValWrapper 兼容 3 种 dataset 类型 (codex Critical #3)."""
    print("\n=== T9: wrapper compat (PACS/Office, MNIST-like, SVHN-like) ===")
    import numpy as _np
    import torch as _t
    from PIL import Image as _Img
    from torchvision import transforms as _T
    from datasets.utils.lab_partition import _DeterministicValWrapper, _extract_targets

    et = _T.Compose([_T.Resize((16, 16)), _T.ToTensor()])
    passed = 0; total = 0

    # T9a: ImageFolder-like (samples + loader 属性)
    class _FakeImageFolder:
        def __init__(self):
            self.samples = [(f"/fake/a/{i}.png", i % 3) for i in range(20)]
            self.targets = [s[1] for s in self.samples]
        def loader(self, path):
            return _Img.fromarray(_np.zeros((32, 32, 3), dtype=_np.uint8), mode="RGB")
    fk = _FakeImageFolder()
    w = _DeterministicValWrapper(fk, indices=[0, 5, 10], eval_transform=et)
    total += 1
    try:
        img, lbl = w[0]
        if isinstance(img, _t.Tensor) and img.shape == (3, 16, 16) and lbl == 0:
            passed += 1; print(f"  ✅ T9a ImageFolder-like: img.shape={tuple(img.shape)}, lbl={lbl}")
        else:
            print(f"  ❌ T9a ImageFolder-like: shape={img.shape}, lbl={lbl}")
    except Exception as e:
        print(f"  ❌ T9a crashed: {e}")
    total += 1
    targets = _extract_targets(fk)
    if len(targets) == 20 and targets[5] == 5 % 3:
        passed += 1; print(f"  ✅ T9a extract_targets: len={len(targets)}, sample={targets[:5].tolist()}")
    else:
        print(f"  ❌ T9a extract_targets: {targets}")

    # T9b: MNIST/USPS-like (data + targets, 单通道 numpy)
    class _FakeMNIST:
        def __init__(self):
            self.data = _t.from_numpy(_np.random.randint(0, 256, size=(20, 8, 8), dtype=_np.uint8))
            self.targets = _t.tensor([i % 10 for i in range(20)])
    fm = _FakeMNIST()
    w2 = _DeterministicValWrapper(fm, indices=[0, 7, 19], eval_transform=et)
    total += 1
    try:
        img, lbl = w2[1]
        if isinstance(img, _t.Tensor) and img.shape == (3, 16, 16) and lbl == 7:
            passed += 1; print(f"  ✅ T9b MNIST-like: img.shape={tuple(img.shape)}, lbl={lbl}")
        else:
            print(f"  ❌ T9b MNIST-like: shape={img.shape}, lbl={lbl}")
    except Exception as e:
        print(f"  ❌ T9b crashed: {e}")
    total += 1
    targets2 = _extract_targets(fm)
    if len(targets2) == 20 and targets2[7] == 7:
        passed += 1; print(f"  ✅ T9b extract_targets: len={len(targets2)}")
    else:
        print(f"  ❌ T9b extract_targets")

    # T9c: SVHN-like (data + labels, (3, H, W, N) numpy)
    class _FakeSVHN:
        def __init__(self):
            # SVHN 真实结构: data 是 (N, 3, H, W) numpy uint8 (按 torchvision SVHN 实现)
            self.data = _np.random.randint(0, 256, size=(20, 3, 8, 8), dtype=_np.uint8)
            self.labels = _np.array([i % 10 for i in range(20)], dtype=_np.int64)
    fs = _FakeSVHN()
    w3 = _DeterministicValWrapper(fs, indices=[0, 5, 19], eval_transform=et)
    total += 1
    try:
        img, lbl = w3[2]
        if isinstance(img, _t.Tensor) and img.shape == (3, 16, 16) and lbl == 19 % 10:
            passed += 1; print(f"  ✅ T9c SVHN-like: img.shape={tuple(img.shape)}, lbl={lbl}")
        else:
            print(f"  ❌ T9c SVHN-like: shape={img.shape}, lbl={lbl}")
    except Exception as e:
        print(f"  ❌ T9c crashed: {e}")
    total += 1
    targets3 = _extract_targets(fs)
    if len(targets3) == 20:
        passed += 1; print(f"  ✅ T9c extract_targets: len={len(targets3)}")
    else:
        print(f"  ❌ T9c extract_targets")

    return passed, total


def t10_lab_state_full_loop():
    """T10: LabState ROI/waste 全链路 (修 codex Important #5)."""
    print("\n=== T10: LabState full loop with acc + clip count ===")
    state = LabState(window_size=5, waste_roi_threshold=0.5)
    sample_share = {"a": 0.5, "b": 0.5}
    passed = 0; total = 0

    # 5 round: a 学得越来越差 (loss 越来越高), b 一直好
    for r in range(1, 11):
        loss_a = 0.5 + r * 0.05  # 升
        loss_b = 0.1
        state.update_val_loss(r, {"a": loss_a, "b": loss_b})
        state.update_test_acc(r, {"a": 50.0 + r, "b": 90.0})
        result = state.compute_lab(round_idx=r + 1, sample_share_dom=sample_share)
        state.update_boost_record(r, sample_share, result)

    # 现在 acc_history 应该有 10 个值
    total += 1
    if len(state.acc_history["a"]) == 10:
        passed += 1; print(f"  ✅ acc_history length: {len(state.acc_history['a'])}")
    else:
        print(f"  ❌ acc_history length: {len(state.acc_history['a'])} (expected 10)")

    # boost_history 应该有 10 个值
    total += 1
    if len(state.boost_history["a"]) == 10 and len(state.boost_history["b"]) == 10:
        passed += 1; print(f"  ✅ boost_history len 10/10")
    else:
        print(f"  ❌ boost_history: a={len(state.boost_history['a'])}, b={len(state.boost_history['b'])}")

    # ratio_history (修复 codex #5)
    total += 1
    if len(state.ratio_history["a"]) == 10:
        passed += 1; print(f"  ✅ ratio_history len 10")
    else:
        print(f"  ❌ ratio_history len: {len(state.ratio_history['a'])}")

    # 计算 ROI
    roi_a = state.compute_window_roi("a", 10)
    roi_a_cum = state.compute_cumulative_roi("a")
    total += 1
    if roi_a is not None and roi_a_cum is not None:
        passed += 1; print(f"  ✅ ROI computable: window_roi={roi_a:.3f}, cum_roi={roi_a_cum:.3f}")
    else:
        print(f"  ❌ ROI: window={roi_a}, cum={roi_a_cum}")

    return passed, total


def t11_imagefolder_custom_path():
    """T11: ImageFolder_Custom 走 rel→abs 转换路径 (修 codex Critical 1, 二轮)."""
    print("\n=== T11: ImageFolder_Custom rel→abs path (PACS/Office) ===")
    import numpy as _np
    import torch as _t
    from PIL import Image as _Img
    from torchvision import transforms as _T
    from datasets.utils.lab_partition import (
        _DeterministicValWrapper, _extract_targets, _get_dataset_size,
    )

    # Mock ImageFolder_Custom: 模拟 PACS/Office 的 dataset 结构
    class _MockImageFolderObj:
        def __init__(self, n_total=30):
            # 30 张总样本, 类别 0-6 (PACS 7 类)
            self.samples = [(f"/fake/{i}.png", i % 7) for i in range(n_total)]
            self.targets = _t.tensor([s[1] for s in self.samples])
        def loader(self, path):
            return _Img.fromarray(_np.zeros((32, 32, 3), dtype=_np.uint8), mode="RGB")

    class _MockImageFolderCustom:
        def __init__(self):
            self.imagefolder_obj = _MockImageFolderObj(n_total=30)
            # train_index_list 是 80% (24 张), test_index_list 是 20% (6 张)
            # 用 i%10 <= 7 模拟 (跟 PACS/Office 实际逻辑一致)
            self.train_index_list = [i for i in range(30) if i % 10 <= 7]
            self.test_index_list = [i for i in range(30) if i % 10 > 7]

    mock_ds = _MockImageFolderCustom()
    et = _T.Compose([_T.Resize((16, 16)), _T.ToTensor()])

    passed = 0; total = 0

    # T11a: detect_extractor 识别 imagefolder_custom
    w = _DeterministicValWrapper(mock_ds, indices=[0, 5], eval_transform=et)
    total += 1
    if w._extractor == "imagefolder_custom":
        passed += 1; print(f"  ✅ T11a detect: imagefolder_custom")
    else:
        print(f"  ❌ T11a detect: got {w._extractor}")

    # T11b: __getitem__ 用 rel→abs 转换
    # rel=0 → abs = train_index_list[0] = 0 → target = 0%7 = 0
    total += 1
    try:
        img, lbl = w[0]
        if lbl == 0 and isinstance(img, _t.Tensor) and img.shape == (3, 16, 16):
            passed += 1; print(f"  ✅ T11b rel0→abs0: lbl={lbl}, img.shape={tuple(img.shape)}")
        else:
            print(f"  ❌ T11b rel0→abs0: lbl={lbl}, shape={img.shape}")
    except Exception as e:
        print(f"  ❌ T11b crashed: {e}")

    # T11c: rel=5 → abs = train_index_list[5] = 5 → target = 5%7 = 5
    total += 1
    try:
        img2, lbl2 = w[1]   # indices[1] = 5
        if lbl2 == 5:
            passed += 1; print(f"  ✅ T11c rel5→abs5: lbl={lbl2}")
        else:
            print(f"  ❌ T11c rel5→abs5: lbl={lbl2}")
    except Exception as e:
        print(f"  ❌ T11c crashed: {e}")

    # T11d: _get_dataset_size 返回 train_index_list 长度 (24)
    total += 1
    n = _get_dataset_size(mock_ds)
    if n == 24:
        passed += 1; print(f"  ✅ T11d size: {n} (= len(train_index_list))")
    else:
        print(f"  ❌ T11d size: {n} (expected 24)")

    # T11e: _extract_targets 返回 train_index_list 长度的 array (按 rel idx 顺序)
    total += 1
    targets = _extract_targets(mock_ds)
    # train_index_list = [0,1,2,3,4,5,6,7, 10,11,...]
    # targets[0] = imagefolder.targets[0] = 0
    # targets[8] = imagefolder.targets[10] = 10%7 = 3
    if (len(targets) == 24
            and targets[0] == 0
            and targets[8] == 3):
        passed += 1; print(f"  ✅ T11e targets: len={len(targets)}, [0]={targets[0]}, [8]={targets[8]}")
    else:
        print(f"  ❌ T11e targets: len={len(targets)}, [0]={targets[0]}, [8]={targets[8]}")

    # T11f: setup_lab_val_loaders end-to-end on 2 mock cli (PACS-style)
    total += 1
    try:
        from datasets.utils.lab_partition import setup_lab_val_loaders
        from torch.utils.data import DataLoader, SubsetRandomSampler
        # 2 cli 各拿 train_index_list 的不同部分 (rel idx)
        # cli 0: rel idx 0..11, cli 1: rel idx 12..23
        cli0_sampler = SubsetRandomSampler(_np.arange(0, 12))
        cli1_sampler = SubsetRandomSampler(_np.arange(12, 20))   # 留 4 张给 val
        # mock dataloader
        class _MockDataLoader:
            def __init__(self, ds, sampler):
                self.dataset = ds
                self.sampler = sampler
            def __iter__(self): yield None  # 不 iterate
        trainloaders = [_MockDataLoader(mock_ds, cli0_sampler),
                        _MockDataLoader(mock_ds, cli1_sampler)]
        train_dataset_list = [mock_ds, mock_ds]
        selected_domain_list = ["photo", "photo"]   # 2 cli 同一 domain
        # 极简 args
        class _Args:
            parti_num = 2
            dataset = "fl_pacs"
            local_batch_size = 4
        val_loaders, val_meta = setup_lab_val_loaders(
            trainloaders, train_dataset_list, selected_domain_list,
            args=_Args(), val_size_per_dom=4, val_per_class=1, val_seed=42,
        )
        # 2 cli 应该共拿到 photo domain 的 4 张 val (从 unused = rel 20..23)
        n_val = sum(val_meta["val_n_per_cli"].values())
        if n_val > 0 and "photo" in val_meta["val_n_per_dom"]:
            passed += 1; print(f"  ✅ T11f setup_lab_val_loaders: n_val={n_val}, "
                               f"per_dom={val_meta['val_n_per_dom']}")
        else:
            print(f"  ❌ T11f end-to-end: val_meta={val_meta}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ T11f crashed: {e}")
    return passed, total


def t12_empty_val_no_pollution():
    """T12: update_val_loss({}) 不污染 has_val_loss; compute_lab 缺 domain fallback (修 codex Critical 2)."""
    print("\n=== T12: empty val + missing domain fallback ===")
    state = LabState()
    passed = 0; total = 0

    # T12a: 初始 has_val_loss=False
    total += 1
    if state.has_val_loss is False:
        passed += 1; print(f"  ✅ T12a init has_val_loss=False")

    # T12b: 空 dict update 后 has_val_loss 仍 False
    state.update_val_loss(0, {})
    total += 1
    if state.has_val_loss is False:
        passed += 1; print(f"  ✅ T12b update_val_loss({{}}) has_val_loss still False")
    else:
        print(f"  ❌ T12b: empty dict polluted has_val_loss")

    # T12c: compute_lab 在 has_val_loss=False 时 fallback
    sample_share = {"a": 0.5, "b": 0.5}
    r1 = state.compute_lab(round_idx=1, sample_share_dom=sample_share)
    total += 1
    if r1.get("fallback_to_fedavg", False) and r1["fallback_reason"] == "round_1_no_val_loss_yet":
        passed += 1; print(f"  ✅ T12c fallback when has_val_loss=False")
    else:
        print(f"  ❌ T12c: r1={r1}")

    # T12d: 部分更新后, compute_lab 缺 domain 也 fallback
    state.update_val_loss(1, {"a": 0.3})   # 只有 a, 没有 b
    sample_share_full = {"a": 0.5, "b": 0.5}
    r2 = state.compute_lab(round_idx=2, sample_share_dom=sample_share_full)
    total += 1
    if (r2.get("fallback_to_fedavg", False)
            and "missing_val_loss_for_domains" in r2.get("fallback_reason", "")):
        passed += 1; print(f"  ✅ T12d missing domain fallback: reason={r2['fallback_reason']}")
    else:
        print(f"  ❌ T12d: r2={r2}")

    # T12e: 完整 update 后, LAB 真的跑
    state.update_val_loss(2, {"a": 0.3, "b": 0.5})
    r3 = state.compute_lab(round_idx=3, sample_share_dom=sample_share_full)
    total += 1
    if (not r3.get("fallback_to_fedavg", False)
            and r3["ratio"]["a"] < 1.0
            and r3["ratio"]["b"] > 1.0):
        passed += 1; print(f"  ✅ T12e LAB active after full update: ratio={{a:{r3['ratio']['a']:.3f}, b:{r3['ratio']['b']:.3f}}}")
    else:
        print(f"  ❌ T12e: r3={r3}")

    return passed, total


def t13_lab_delta_acc_patch():
    """T13: 端到端调用 F2DCPgLab.lab_record_test_acc, 验证 proto_logs[-1].lab_delta_acc patch
    (codex 四轮 Minor #3: 之前是复制 patch 逻辑, 不是直接调 model method).

    用 fake model 持有 lab_state + proto_logs, 直接调 unbound method 走真实代码路径.
    """
    print("\n=== T13: lab_record_test_acc end-to-end (real method call) ===")
    from models.f2dc_pg_lab import F2DCPgLab
    passed = 0; total = 0

    # Fake minimal model (避免 instantiate 整个 F2DCPgLab, 因为它需要 nets_list)
    class _FakeModel:
        def __init__(self):
            self.lab_state = LabState()
            self.proto_logs = []
            self.lab_warn_interval = 0   # 不触发 waste warning (T15 单独测)
            self._next_round_lab_result = None
            self.lab_print_diag = False

    fake = _FakeModel()

    # ===== Round 1: 模拟 loc_update 写入 滞后值, evaluate 后 patch =====
    fake.proto_logs.append({"round": 0, "lab_delta_acc_a": -999.0, "lab_delta_acc_b": -999.0})
    F2DCPgLab.lab_record_test_acc(fake, round_idx=1, accs=[50.0, 60.0],
                                   all_dataset_names=["a", "b"])
    latest = fake.proto_logs[-1]
    total += 1
    # acc_history 只有 1 个值, delta = 0
    if latest["lab_delta_acc_a"] == 0.0 and latest["lab_delta_acc_b"] == 0.0:
        passed += 1; print(f"  ✅ T13a R1 (无前一轮): delta_acc 全 0 (real method call)")
    else:
        print(f"  ❌ T13a: {latest}")

    # ===== Round 2: 真实 delta_acc patch =====
    fake.proto_logs.append({"round": 1, "lab_delta_acc_a": -999.0, "lab_delta_acc_b": -999.0})
    F2DCPgLab.lab_record_test_acc(fake, round_idx=2, accs=[55.0, 70.0],
                                   all_dataset_names=["a", "b"])
    latest = fake.proto_logs[-1]
    total += 1
    # acc_history[a] = [50, 55], delta = +5; acc_history[b] = [60, 70], delta = +10
    if (abs(latest["lab_delta_acc_a"] - 5.0) < 1e-9
            and abs(latest["lab_delta_acc_b"] - 10.0) < 1e-9):
        passed += 1; print(f"  ✅ T13b R2 patch: a=+5.0, b=+10.0 (real method call)")
    else:
        print(f"  ❌ T13b: {latest}")

    # ===== Round 3: 退步 =====
    fake.proto_logs.append({"round": 2, "lab_delta_acc_a": -999.0, "lab_delta_acc_b": -999.0})
    F2DCPgLab.lab_record_test_acc(fake, round_idx=3, accs=[53.0, 65.0],
                                   all_dataset_names=["a", "b"])
    latest = fake.proto_logs[-1]
    total += 1
    # acc_history = [50, 55, 53] / [60, 70, 65]
    if (abs(latest["lab_delta_acc_a"] - (-2.0)) < 1e-9
            and abs(latest["lab_delta_acc_b"] - (-5.0)) < 1e-9):
        passed += 1; print(f"  ✅ T13c R3 negative delta: a=-2.0, b=-5.0 (real method call)")
    else:
        print(f"  ❌ T13c: {latest}")

    # ===== T13d: acc_history 同步增加 =====
    total += 1
    if (len(fake.lab_state.acc_history["a"]) == 3
            and fake.lab_state.acc_history["a"] == [50.0, 55.0, 53.0]):
        passed += 1; print(f"  ✅ T13d acc_history populated correctly")
    else:
        print(f"  ❌ T13d: acc_history={fake.lab_state.acc_history}")

    return passed, total


def t15_waste_warning_post_eval():
    """T15: live waste warning 在 evaluate 后才打印 (codex 四轮 Important).

    验证 _print_waste_warnings 用 round_idx_override 收到正确的 round 计数,
    且 detect_waste 用最新 acc_history.
    """
    print("\n=== T15: waste warning post-eval (codex 四轮) ===")
    from models.f2dc_pg_lab import F2DCPgLab
    passed = 0; total = 0

    class _FakeModel:
        # 借调 unbound method 给 fake 用 (Python descriptor 自动 bind self=fake)
        _print_waste_warnings = F2DCPgLab._print_waste_warnings

        def __init__(self):
            self.lab_state = LabState(window_size=3, waste_roi_threshold=0.5)
            self.proto_logs = []
            self.lab_warn_interval = 2   # 每 2 round 检查
            # 模拟一个固定 LAB result (本 round 给 a 加成)
            self._next_round_lab_result = {
                "domains": ["a", "b"],
                "fallback_to_fedavg": False,
            }
            self.lab_print_diag = False
            self.epoch_index = 0

    fake = _FakeModel()
    # Pre-fill LabState with boost + acc data 模拟过去 5 round
    sample_share = {"a": 0.5, "b": 0.5}
    for r in range(1, 6):
        fake.lab_state.update_val_loss(r, {"a": 0.5, "b": 0.1})
        # 给 a 加 boost 但 acc 不涨 (浪费 case)
        fake.lab_state.boost_history["a"].append(0.05)
        fake.lab_state.boost_history["b"].append(0.0)
        fake.lab_state.update_test_acc(r, {"a": 50.0, "b": 90.0})   # a acc 不动

    # 现在 round 6 evaluate, lab_record_test_acc 内会调 _print_waste_warnings(round_idx_override=6)
    # 6 % 2 == 0 → 应该触发
    import io, contextlib
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        F2DCPgLab.lab_record_test_acc(fake, round_idx=6, accs=[50.0, 90.0],
                                       all_dataset_names=["a", "b"])
    output = capture.getvalue()
    total += 1
    if "[LAB WASTE WARN R" in output and "dom=a" in output:
        passed += 1; print(f"  ✅ T15a waste warn 触发 + dom=a 标识")
    else:
        print(f"  ❌ T15a output: {output!r}")

    # T15b: round 5 (奇数), 5 % 2 != 0, 不应触发
    capture2 = io.StringIO()
    fake.lab_state.boost_history["a"].append(0.05)
    fake.lab_state.boost_history["b"].append(0.0)
    with contextlib.redirect_stdout(capture2):
        F2DCPgLab.lab_record_test_acc(fake, round_idx=5, accs=[50.0, 90.0],
                                       all_dataset_names=["a", "b"])
    output2 = capture2.getvalue()
    total += 1
    if "[LAB WASTE WARN" not in output2:
        passed += 1; print(f"  ✅ T15b R5 (奇数): waste warn 不触发")
    else:
        print(f"  ❌ T15b: 不该触发 but got {output2!r}")
    return passed, total


def t14_pacs_val_size_35():
    """T14: PACS 7 类 × per_class=5 → 单域 val cap=35 (codex 三轮 Important #2)."""
    print("\n=== T14: PACS val_size 35 (C=7 × per_class=5) ===")
    import numpy as _np
    from datasets.utils.lab_partition import _stratified_sample_indices

    rng = _np.random.RandomState(42)
    # 模拟 PACS 100 张 unused, 7 类
    candidate = _np.arange(100)
    targets = _np.array([i % 7 for i in range(100)])
    # 用户传 val_size_per_dom=50, per_class=5 → 实际 cap = min(50, 7*5) = 35
    selected, class_counts = _stratified_sample_indices(
        candidate_idx=candidate,
        targets_for_candidates=targets,
        per_class=5,
        max_total=50,
        rng=rng,
    )
    passed = 0; total = 0
    total += 1
    if len(selected) == 35 and all(c == 5 for c in class_counts.values()) and len(class_counts) == 7:
        passed += 1; print(f"  ✅ T14a PACS-like: total={len(selected)} (cap by 7×5=35), all classes 5")
    else:
        print(f"  ❌ T14a: total={len(selected)}, class_counts={class_counts}")

    # Office C=10, per_class=5, max_total=50 → cap=50
    targets10 = _np.array([i % 10 for i in range(100)])
    selected2, cc2 = _stratified_sample_indices(
        candidate_idx=_np.arange(100),
        targets_for_candidates=targets10,
        per_class=5,
        max_total=50,
        rng=rng,
    )
    total += 1
    if len(selected2) == 50 and all(c == 5 for c in cc2.values()) and len(cc2) == 10:
        passed += 1; print(f"  ✅ T14b Office-like: total={len(selected2)} (=50, full)")
    else:
        print(f"  ❌ T14b: total={len(selected2)}, cc={cc2}")
    return passed, total


def t8_f2dc_pg_lab_import():
    """T8: F2DCPgLab class 可 import + NAME / 类继承层级正确."""
    print("\n=== T8: F2DCPgLab importability ===")
    passed = 0; total = 0
    try:
        from models.f2dc_pg_lab import F2DCPgLab
        from models.f2dc_pg import F2DCPG
        total += 1
        if F2DCPgLab.NAME == "f2dc_pg_lab":
            passed += 1; print(f"  ✅ NAME: {F2DCPgLab.NAME}")
        else:
            print(f"  ❌ NAME: {F2DCPgLab.NAME}")
        total += 1
        if issubclass(F2DCPgLab, F2DCPG):
            passed += 1; print(f"  ✅ Inheritance: F2DCPgLab subclass of F2DCPG")
        else:
            print(f"  ❌ Not subclass of F2DCPG")
        # 关键 method 存在
        for m in ["loc_update", "aggregate_nets", "_ensure_lab_setup",
                  "_compute_sample_share_dom", "_domain_level_to_cli_freq"]:
            total += 1
            if hasattr(F2DCPgLab, m):
                passed += 1; print(f"  ✅ method: {m}")
            else:
                print(f"  ❌ missing method: {m}")
    except Exception as e:
        total += 1
        print(f"  ❌ Import failed: {e}")
    return passed, total


# =====================================================
def main():
    print("=" * 70)
    print("  LAB v4.2 Sanity Tests")
    print("=" * 70)

    all_passed = 0; all_total = 0
    for fn in [t1_projection_corners, t2_gap_zero_fedavg, t3_pacs_realistic,
               t4_labstate_ema, t5_labstate_compute_lab,
               t6_lab_partition_stratify, t7_eval_transform_deterministic,
               t8_f2dc_pg_lab_import,
               t9_wrapper_compat, t10_lab_state_full_loop,
               t11_imagefolder_custom_path, t12_empty_val_no_pollution,
               t13_lab_delta_acc_patch, t14_pacs_val_size_35,
               t15_waste_warning_post_eval]:
        try:
            p, t = fn()
            all_passed += p; all_total += t
        except Exception as e:
            print(f"  ❌ {fn.__name__} crashed: {e}")
            all_total += 1
            import traceback; traceback.print_exc()

    print(f"\n{'=' * 70}\n  TOTAL: {all_passed}/{all_total} passed\n{'=' * 70}")
    return 0 if all_passed == all_total else 1


if __name__ == "__main__":
    sys.exit(main())
