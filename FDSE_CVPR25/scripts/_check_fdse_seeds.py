"""Check FDSE Office single-seed data + locate BiProto s=333 best ckpt."""
import json, glob, os

# 1. FDSE Office single-seed data
print("=== FDSE Office records ===")
files = sorted(glob.glob("/home/lry/code/federated-learning/FDSE_CVPR25/task/office_caltech10_c4/record/fdse_*R200*.json"))
print(f"FDSE Office runs found: {len(files)}")
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    a = d.get("mean_local_test_accuracy", [])
    if not a:
        continue
    a = [x * 100 if x < 2 else x for x in a]
    seed = "?"
    for s in ["S2", "S15", "S333", "S42"]:
        if f"_{s}_" in f:
            seed = s
            break
    best = max(a)
    best_r = a.index(best) + 1
    last = a[-1]
    print(f"  FDSE {seed}: best={best:.2f}@R{best_r}, last={last:.2f}")

# 2. Locate BiProto s=333 Office best ckpt
print("\n=== BiProto Office s=333 best ckpt search ===")
ckpts = sorted([d for d in os.listdir("/home/lry/fl_checkpoints/") if "R200_best" in d])
print(f"Total R200 ckpts: {len(ckpts)}")
for d in ckpts:
    full_meta = f"/home/lry/fl_checkpoints/{d}/meta.json"
    if not os.path.isfile(full_meta):
        continue
    m = json.load(open(full_meta))
    seed = m.get("seed")
    nc = m.get("num_clients")
    acc = m.get("best_avg_acc", 0)
    rd = m.get("best_round")
    if seed == 333 and nc == 4 and acc > 90:
        gm = f"/home/lry/fl_checkpoints/{d}/global_model.pt"
        if os.path.isfile(gm):
            import torch
            sd = torch.load(gm, map_location="cpu")
            num_classes = sd.get("head.weight", None)
            if num_classes is not None:
                nc_pred = num_classes.shape[0]
                ds = "Office" if nc_pred == 10 else f"PACS({nc_pred})"
                # check encoder_sty (BiProto) vs orth_only
                is_biproto = "encoder_sty.net.0.weight" in sd
                tag = "BiProto" if is_biproto else "orth_only"
                print(f"  {d}: best={acc:.2f}@R{rd} dataset={ds} algo={tag}")
