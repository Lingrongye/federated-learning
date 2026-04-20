"""从 flgo log 倒推 record JSON.

当 record JSON 因文件名过长 (Errno 36) 或其他 I/O 错误保存失败时, 扫 log 重建 20 个标量指标
序列的兼容 JSON. per-client `*_dist` 字段因 flgo log_once 不打印 list 而无法恢复.

用法:
    python rescue_record_from_log.py \
        --log <path/to/flgo.log> \
        --out <path/to/rescued.json> \
        --config <path/to/yml> \
        [--task office_caltech10_c4] [--seed 2] [--template <path/to/similar_record.json>]

template (可选) 用于继承 option 字段. 若不提供, option 从 config.yml + 默认值构造.
"""
import argparse
import json
import re
from pathlib import Path

SCALAR_METRICS = [
    "local_val_accuracy", "mean_local_val_accuracy", "std_local_val_accuracy",
    "min_local_val_accuracy", "max_local_val_accuracy",
    "local_val_loss", "mean_local_val_loss", "std_local_val_loss",
    "min_local_val_loss", "max_local_val_loss",
    "local_test_accuracy", "mean_local_test_accuracy", "std_local_test_accuracy",
    "min_local_test_accuracy", "max_local_test_accuracy",
    "local_test_loss", "mean_local_test_loss", "std_local_test_loss",
    "min_local_test_loss", "max_local_test_loss",
]

LOG_LINE_RE = re.compile(r"INFO\s+(\w+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(log_path):
    """返回 {metric_name: [round0_val, round1_val, ...]}"""
    series = {m: [] for m in SCALAR_METRICS}
    current_round = {}
    eval_count = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key not in series:
                continue
            current_round[key] = float(val)
            if key == "max_local_test_loss":
                for metric in SCALAR_METRICS:
                    series[metric].append(current_round.get(metric, float("nan")))
                current_round = {}
                eval_count += 1
    return series, eval_count


def build_option(config_path, task, seed, num_rounds):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    option = {
        "sample": "uniform",
        "aggregate": "other",
        "num_rounds": cfg.get("num_rounds", num_rounds),
        "proportion": cfg.get("proportion", 1.0),
        "learning_rate_decay": cfg.get("learning_rate_decay", 0.9998),
        "lr_scheduler": str(cfg.get("lr_scheduler", 0)),
        "early_stop": -1,
        "num_epochs": cfg.get("num_epochs", 1),
        "num_steps": -1,
        "learning_rate": cfg.get("learning_rate", 0.05),
        "batch_size": float(cfg.get("batch_size", 50)),
        "optimizer": "SGD",
        "clip_grad": float(cfg.get("clip_grad", 10)),
        "momentum": 0.0,
        "weight_decay": cfg.get("weight_decay", 1e-3),
        "num_edge_rounds": 5,
        "algo_para": cfg.get("algo_para", []),
        "train_holdout": cfg.get("train_holdout", 0.2),
        "test_holdout": 0.0,
        "local_test": cfg.get("local_test", True),
        "local_test_ratio": 0.5,
        "seed": seed,
        "dataseed": seed,
        "task": f"task/{task}",
        "algorithm": "feddsa_sgpa",
        "model": "feddsa_sgpa",
        "scene": "horizontal",
        "simulator": "Simulator",
    }
    return option


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="flgo log path")
    ap.add_argument("--out", required=True, help="output record JSON path")
    ap.add_argument("--config", help="config yml path (for option field)")
    ap.add_argument("--template", help="optional existing record JSON to inherit option")
    ap.add_argument("--task", default="office_caltech10_c4")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    print(f"[rescue] scanning {log_path}")
    series, eval_count = parse_log(log_path)
    lengths = {k: len(v) for k, v in series.items()}
    if len(set(lengths.values())) != 1:
        print("[warn] metric lengths differ:", lengths)
    n_rounds = max(lengths.values())
    print(f"[rescue] extracted {eval_count} eval checkpoints, {n_rounds} round entries per metric")

    if args.template:
        with open(args.template, "r", encoding="utf-8") as f:
            template = json.load(f)
        option = template.get("option", {})
        option["seed"] = args.seed
        option["dataseed"] = args.seed
    elif args.config:
        option = build_option(args.config, args.task, args.seed, n_rounds - 1)
    else:
        option = {"num_rounds": n_rounds - 1, "seed": args.seed, "task": f"task/{args.task}"}

    record = {"option": option}
    for metric in SCALAR_METRICS:
        record[metric] = series[metric]
    # per-client dist unrecoverable -> empty
    for dist_key in [
        "local_val_accuracy_dist", "local_val_loss_dist",
        "local_test_accuracy_dist", "local_test_loss_dist",
    ]:
        record[dist_key] = []

    record["_rescue_meta"] = {
        "rescued_from_log": True,
        "log_source": str(log_path),
        "n_eval_checkpoints": eval_count,
        "missing_fields": [
            "local_val_accuracy_dist",
            "local_val_loss_dist",
            "local_test_accuracy_dist",
            "local_test_loss_dist",
        ],
        "note": "per-client dist lists unrecoverable (flgo log_once skips list fields). "
        "Use min/max_* to bound worst/best client.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    print(f"[rescue] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    avg = record["mean_local_test_accuracy"]
    if avg:
        best = max(avg)
        best_r = avg.index(best)
        print(f"[rescue] sanity: AVG max={best*100:.2f}@R{best_r}, last={avg[-1]*100:.2f}")


if __name__ == "__main__":
    main()
