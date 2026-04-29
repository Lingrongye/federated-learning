"""
Leave-one-out cos sim 重算 + 从 F2DC heavy dump features 反推 per-client per-class proto baseline.

排除两个 bias:
1. self-inclusion bias: consensus 里包含 client 自己 → 用 LOO 排除
2. backbone-shared bias: 同 backbone 提 feature 本来就该相似 → 用 F2DC (没 cross-attention) 当 baseline
"""
import os
import numpy as np
from glob import glob

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office"
DOMAINS = ["caltech", "amazon", "webcam", "dslr"]


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def proto_cos_loo_traj(diag_dir):
    """
    leave-one-out cos sim trajectory.
    Returns rs (R,), traj (R, K) = client_k 跟 mean(others) 的 cos sim, 按 class 平均.
    """
    files = sorted(glob(os.path.join(diag_dir, "round_*.npz")))
    rs, traj_loo, traj_self = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "local_protos" not in d.files:
            return None, None, None
        rs.append(int(d["round"]))
        lp = d["local_protos"].astype(np.float32)  # (K, C, dim)
        K, C, dim = lp.shape
        total = lp.sum(axis=0)  # (C, dim) — 含全部 client
        cos_loo_per_client = []
        cos_self_per_client = []
        for k in range(K):
            others = (total - lp[k]) / max(K - 1, 1)  # (C, dim) — 排除自己
            consensus_self = total / K                 # 旧版: 含自己
            sim_loo, sim_self = [], []
            for c in range(C):
                if np.linalg.norm(lp[k, c]) < 1e-6:
                    continue  # client_k 此 round 此 class 无样本
                if np.linalg.norm(others[c]) >= 1e-6:
                    sim_loo.append(cos(lp[k, c], others[c]))
                if np.linalg.norm(consensus_self[c]) >= 1e-6:
                    sim_self.append(cos(lp[k, c], consensus_self[c]))
            cos_loo_per_client.append(np.mean(sim_loo) if sim_loo else np.nan)
            cos_self_per_client.append(np.mean(sim_self) if sim_self else np.nan)
        traj_loo.append(cos_loo_per_client)
        traj_self.append(cos_self_per_client)
    return np.array(rs), np.array(traj_loo), np.array(traj_self)


def reconstruct_f2dc_proto_from_heavy(heavy):
    """
    从 F2DC heavy dump 的 features 反推 per-domain per-class mean (= per-client per-class proto for office,
    因为 office 的 client_i 跟 domain_i 是一一对应的 — 但实际 office 是 10 client 跨 4 domain).

    这里降级: 按 domain 算 per-class proto (4 个 domain × 10 类 mean), 当 4 "client" 用.
    """
    if heavy is None:
        return None, None
    feats_d = heavy["features"].item()
    labels_d = heavy["labels"].item()
    K = len(DOMAINS)
    # 先确定 C
    all_labels = np.concatenate([labels_d[d] for d in DOMAINS if d in labels_d])
    C = int(all_labels.max() + 1)
    dim = feats_d[DOMAINS[0]].shape[1]
    proto = np.zeros((K, C, dim), dtype=np.float32)
    valid = np.zeros((K, C), dtype=bool)
    for k, dn in enumerate(DOMAINS):
        if dn not in feats_d:
            continue
        feats = feats_d[dn].astype(np.float32)
        labs = labels_d[dn].astype(np.int32)
        for c in range(C):
            mask = labs == c
            if mask.sum() > 0:
                proto[k, c] = feats[mask].mean(axis=0)
                valid[k, c] = True
    return proto, valid


def cos_matrix_loo(proto, valid):
    """
    proto: (K, C, dim), valid: (K, C) bool.
    Returns avg cos sim (over valid classes) for each k vs mean(others).
    Also returns mean (含 self) for fairness.
    """
    K, C, dim = proto.shape
    cos_loo, cos_self = [], []
    for k in range(K):
        sim_loo, sim_self = [], []
        for c in range(C):
            if not valid[k, c]:
                continue
            other_mask = valid[:, c].copy()
            other_mask[k] = False
            if other_mask.sum() == 0:
                continue
            others = proto[other_mask, c].mean(axis=0)
            consensus_self = proto[valid[:, c], c].mean(axis=0)
            sim_loo.append(cos(proto[k, c], others))
            sim_self.append(cos(proto[k, c], consensus_self))
        cos_loo.append(np.mean(sim_loo) if sim_loo else np.nan)
        cos_self.append(np.mean(sim_self) if sim_self else np.nan)
    return np.array(cos_loo), np.array(cos_self)


def main():
    print("=" * 90)
    print("Leave-one-out cos sim 重算 + F2DC baseline 对照")
    print("=" * 90)

    # 1. PG-DFC 系列: LOO trajectory
    print("\n【PG-DFC 系列 — local_protos 直接 LOO】")
    print("-" * 90)
    print(f"{'method':18s} {'seed':>4s} | r1     r10    r50    r100  | LOO @ R100  | self-inc @ R100 | drop")
    pg_runs = [
        ("PG-DFC", 2, "diag_pgdfc_office_s2"),
        ("PG-DFC", 15, "diag_pgdfc_office_s15"),
        ("PG-DFC+DaA", 2, "diag_pgdfc_daa_office_s2"),
        ("PG-DFC+DaA", 15, "diag_pgdfc_daa_office_s15"),
        ("PG-DFC+DaA", 333, "diag_pgdfc_daa_office_s333"),
    ]
    pg_results = {}
    for method, seed, sub in pg_runs:
        d = os.path.join(ROOT, sub)
        rs, traj_loo, traj_self = proto_cos_loo_traj(d)
        if rs is None:
            continue
        avg_loo = np.nanmean(traj_loo, axis=1)
        avg_self = np.nanmean(traj_self, axis=1)
        idx = lambda r: min(r - 1, len(rs) - 1)
        r1, r10, r50, r100 = avg_loo[idx(1)], avg_loo[idx(10)], avg_loo[idx(50)], avg_loo[-1]
        self100 = avg_self[-1]
        drop = self100 - r100
        print(f"  {method:16s} {seed:4d} | {r1:.3f}  {r10:.3f}  {r50:.3f}  {r100:.3f} |   {r100:.3f}     |    {self100:.3f}      | -{drop:.3f}")
        pg_results[(method, seed)] = traj_loo

    # 2. PG-DFC+DaA per-client LOO breakdown @ R100
    print("\n【PG-DFC+DaA seed=2 — per-client LOO @ R100 (排除 self-inclusion 后)】")
    print("-" * 90)
    d = os.path.join(ROOT, "diag_pgdfc_daa_office_s2")
    z = np.load(os.path.join(d, "round_001.npz"), allow_pickle=True)
    dom_per_client = list(z["domain_per_client"])
    rs, traj_loo, traj_self = proto_cos_loo_traj(d)
    last_loo = traj_loo[-1]
    last_self = traj_self[-1]
    print(f"{'client':>8s} {'domain':>8s} | LOO     self-inc  | drop")
    for k in range(10):
        print(f"  {k:6d}  {dom_per_client[k]:>8s} | {last_loo[k]:.3f}    {last_self[k]:.3f}     | -{last_self[k]-last_loo[k]:.3f}")

    # 3. F2DC baseline: 从 heavy dump features 反推 per-domain proto
    print("\n【F2DC 系列 baseline — 从 heavy dump features 反推 per-domain proto, 算 LOO cos sim】")
    print("(F2DC 没用 cross-attention 拉 feature, 这个数字 = '同 backbone + 同 class 的天然水平')")
    print("-" * 90)
    print(f"{'method':18s} {'seed':>4s} {'snap':>5s} | LOO    self-inc")
    f2dc_runs = [
        ("F2DC", 2, "diag_f2dc_office_s2"),
        ("F2DC", 15, "diag_f2dc_office_s15"),
        ("F2DC+DaA", 2, "diag_f2dc_daa_office_s2"),
        ("F2DC+DaA", 15, "diag_f2dc_daa_office_s15"),
        ("F2DC+DaA", 333, "diag_f2dc_daa_office_s333"),
    ]
    f2dc_baseline = {}
    for method, seed, sub in f2dc_runs:
        d = os.path.join(ROOT, sub)
        for snap in ["best", "final"]:
            files = sorted(glob(os.path.join(d, f"{snap}_*.npz")))
            if not files:
                continue
            heavy = np.load(files[0] if snap == "best" else files[-1], allow_pickle=True)
            proto, valid = reconstruct_f2dc_proto_from_heavy(heavy)
            cos_loo, cos_self = cos_matrix_loo(proto, valid)
            mean_loo = np.nanmean(cos_loo)
            mean_self = np.nanmean(cos_self)
            print(f"  {method:16s} {seed:4d} {snap:>5s} | {mean_loo:.3f}    {mean_self:.3f}")
            f2dc_baseline[(method, seed, snap)] = mean_loo

    # 4. PG-DFC heavy dump 也反推一下 (跟 F2DC 同方法对照)
    print("\n【PG-DFC 系列 — 从 heavy dump features 反推 per-domain proto, 算 LOO cos sim】")
    print("(用同样方法跟 F2DC 比, 排除 'feature dim / proto 计算方式' 差异)")
    print("-" * 90)
    print(f"{'method':18s} {'seed':>4s} {'snap':>5s} | LOO    self-inc")
    pg_baseline = {}
    for method, seed, sub in pg_runs:
        d = os.path.join(ROOT, sub)
        for snap in ["best", "final"]:
            files = sorted(glob(os.path.join(d, f"{snap}_*.npz")))
            if not files:
                continue
            heavy = np.load(files[0] if snap == "best" else files[-1], allow_pickle=True)
            proto, valid = reconstruct_f2dc_proto_from_heavy(heavy)
            cos_loo, cos_self = cos_matrix_loo(proto, valid)
            mean_loo = np.nanmean(cos_loo)
            mean_self = np.nanmean(cos_self)
            print(f"  {method:16s} {seed:4d} {snap:>5s} | {mean_loo:.3f}    {mean_self:.3f}")
            pg_baseline[(method, seed, snap)] = mean_loo

    # 5. 对照表 — 同方法对照 (heavy reconstruct)
    print("\n" + "=" * 90)
    print("⭐ 核心对照表 (heavy reconstruct, per-domain proto, LOO cos sim @ final R100)")
    print("=" * 90)
    print(f"{'seed':>4s} | F2DC   F2DC+DaA  PG-DFC  PG-DFC+DaA | Δ(PG+DaA - F2DC+DaA)")
    for seed in [2, 15]:
        f2dc = f2dc_baseline.get(("F2DC", seed, "final"), np.nan)
        f2dc_daa = f2dc_baseline.get(("F2DC+DaA", seed, "final"), np.nan)
        pgdfc = pg_baseline.get(("PG-DFC", seed, "final"), np.nan)
        pgdfc_daa = pg_baseline.get(("PG-DFC+DaA", seed, "final"), np.nan)
        delta = pgdfc_daa - f2dc_daa
        print(f"{seed:4d} | {f2dc:.3f}   {f2dc_daa:.3f}    {pgdfc:.3f}   {pgdfc_daa:.3f}      | {delta:+.3f}")

    # 6. 跨类 mode collapse 对照
    print("\n【Mode collapse: 不同 class 之间的 pairwise cos sim (heavy reconstruct, R100)】")
    print("(不同 class 的 proto 不应相似. 高 = 模型类别区分能力差)")
    print("-" * 90)
    print(f"{'method':18s} {'seed':>4s} | mean_pairwise_class_cos | (越接近 0 越好)")
    for method, seed, sub in pg_runs + f2dc_runs:
        d = os.path.join(ROOT, sub)
        files = sorted(glob(os.path.join(d, "final_*.npz")))
        if not files:
            continue
        heavy = np.load(files[-1], allow_pickle=True)
        proto, valid = reconstruct_f2dc_proto_from_heavy(heavy)
        # average across 4 domain
        K, C, dim = proto.shape
        all_pair = []
        for k in range(K):
            valid_c = np.where(valid[k])[0]
            if len(valid_c) < 2:
                continue
            for i in range(len(valid_c)):
                for j in range(i + 1, len(valid_c)):
                    a, b = proto[k, valid_c[i]], proto[k, valid_c[j]]
                    all_pair.append(cos(a, b))
        mean_pair = np.mean(all_pair) if all_pair else np.nan
        print(f"  {method:16s} {seed:4d} | {mean_pair:.3f}")


if __name__ == "__main__":
    main()
