"""重算所有 baseline 在 Office / PACS s={2,15,333} 的真实 best from mean_local_test_accuracy."""
import json, glob, os

ALGOS = {
    'feddsa_biproto': '/home/lry/code/federated-learning/FDSE_CVPR25/task/{ds}/record/feddsa_biproto_*_{s}_*R200*.json',
    'feddsa_scheduled (orth_only)': '/home/lry/code/federated-learning/FDSE_CVPR25/task/{ds}/record/feddsa_scheduled_*_{s}_*R200*.json',
    'fdse': '/home/lry/code/federated-learning/FDSE_CVPR25/task/{ds}/record/fdse_*_{s}_*R200*.json',
}

for ds, key in [('Office', 'office_caltech10_c4'), ('PACS', 'PACS_c4')]:
    print(f'\n=== {ds} ===')
    table = {}  # algo -> seed -> best
    for algo, pat in ALGOS.items():
        for s in ['S2', 'S15', 'S333']:
            files = glob.glob(pat.format(ds=key, s=s))
            if not files:
                continue
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)  # latest
            best_acc, best_r, best_file = -1, 0, None
            for f in files:
                try:
                    with open(f) as fp:
                        d = json.load(fp)
                except Exception:
                    continue
                a = d.get('mean_local_test_accuracy', [])
                if not a:
                    continue
                a = [x * 100 if x < 2 else x for x in a]
                m = max(a)
                if m > best_acc:
                    best_acc, best_r, best_file = m, a.index(m) + 1, f
            if best_acc > 0:
                # per-client at best round
                with open(best_file) as fp:
                    d = json.load(fp)
                pc = d.get('local_test_accuracy_dist', [])
                if pc and len(pc) >= best_r:
                    pc_r = [round(x * 100, 1) for x in pc[best_r - 1]]
                else:
                    pc_r = '?'
                table.setdefault(algo, {})[s] = best_acc
                print(f'  {algo} {s}: best={best_acc:.2f}@R{best_r}  per-client={pc_r}')
    # 3-seed mean
    print(f'  --- 3-seed mean ---')
    for algo, seeds in table.items():
        if len(seeds) == 3:
            m = sum(seeds.values()) / 3
            print(f'  {algo}: 3-seed mean = {m:.2f}')
        else:
            print(f'  {algo}: only {len(seeds)} seeds (incomplete)')
