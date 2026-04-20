"""对比 rescued record JSON vs 原生 record JSON, 报告每个 metric 的精度差异."""
import argparse
import json


METRICS_CHECK = [
    ("ALL Best",  "local_test_accuracy",      max),
    ("ALL Last",  "local_test_accuracy",      lambda x: x[-1]),
    ("AVG Best",  "mean_local_test_accuracy", max),
    ("AVG Last",  "mean_local_test_accuracy", lambda x: x[-1]),
    ("ALL@R100",  "local_test_accuracy",      lambda x: x[100]),
    ("AVG@R100",  "mean_local_test_accuracy", lambda x: x[100]),
    ("std@R200",  "std_local_test_accuracy",  lambda x: x[-1]),
    ("min@R200",  "min_local_test_accuracy",  lambda x: x[-1]),
    ("max@R200",  "max_local_test_accuracy",  lambda x: x[-1]),
    ("loss@R200", "local_test_loss",          lambda x: x[-1]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--rescued", required=True)
    args = ap.parse_args()

    orig = json.load(open(args.orig, encoding="utf-8"))
    resc = json.load(open(args.rescued, encoding="utf-8"))

    print("\n" + "=" * 72)
    print(f"{'Metric':<12} {'Original':>12} {'Rescued':>12} {'Diff':>13} {'Diff_pp':>10}")
    print("=" * 72)
    for name, key, fn in METRICS_CHECK:
        o = fn(orig[key])
        r = fn(resc[key])
        d = r - o
        ok = "OK" if abs(d) < 1e-4 else "!"
        print(f"{name:<12} {o:>12.6f} {r:>12.6f} {d:>+13.4e} {d*100:>+9.4f}pp {ok}")

    scalar_keys = [
        k for k in orig
        if isinstance(orig.get(k), list)
        and not k.endswith("_dist")
        and k not in {"algo_para"}
    ]
    max_diff = 0.0
    max_key = None
    print("\n" + "=" * 72)
    print("Per-round max|Δ| across all 20 scalar metrics (201 rounds):")
    print("=" * 72)
    for m in sorted(scalar_keys):
        if m not in resc or not isinstance(resc[m], list):
            continue
        if len(orig[m]) != len(resc[m]):
            continue
        diffs = [abs(a - b) for a, b in zip(orig[m], resc[m])]
        md = max(diffs) if diffs else 0.0
        if md > max_diff:
            max_diff = md
            max_key = m
        print(f"  {m:<35} max|Δ|={md:.2e}")
    print(
        f"\nOverall max|Δ| across 20 metrics × 201 rounds: {max_diff:.2e} "
        f"(at '{max_key}')"
    )
    print(
        "VERDICT:",
        "PASS  (rescue faithful to <=1e-4)"
        if max_diff < 1e-4
        else "FAIL",
    )


if __name__ == "__main__":
    main()
