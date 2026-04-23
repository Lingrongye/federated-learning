import numpy as np

def analyze(path, name):
    vals = [float(l.strip()) for l in open(path) if l.strip()]
    vals = np.array(vals)
    n = len(vals)
    best = vals.max()
    best_round = int(vals.argmax())
    last10 = vals[-10:] if n >= 10 else vals

    # Stability: std of last 10 rounds
    stability = float(last10.std())

    # Convergence: where did it first reach 90% of best?
    threshold = best * 0.9
    conv_round = next((i for i, v in enumerate(vals) if v >= threshold), n)

    # Drops: count how many times acc dropped > 5% from previous
    drops = sum(1 for i in range(1, n) if vals[i] < vals[i-1] - 0.05)

    # Oscillation: mean absolute change in last 50 rounds
    last50 = vals[-50:] if n >= 50 else vals
    oscillation = float(np.abs(np.diff(last50)).mean()) if len(last50) > 1 else 0

    print("{}:".format(name))
    print("  Rounds logged: {}".format(n))
    print("  Best acc: {:.4f} (round {})".format(best, best_round))
    print("  Last acc: {:.4f}".format(vals[-1]))
    print("  Last 10 mean: {:.4f}  std: {:.4f}".format(last10.mean(), stability))
    print("  90% convergence round: {}".format(conv_round))
    print("  Major drops (>5%): {}".format(drops))
    print("  Last 50 range: [{:.4f}, {:.4f}]".format(last50.min(), last50.max()))
    print("  Last 50 oscillation (mean abs change): {:.4f}".format(oscillation))
    print()

analyze("/tmp/feddsa_acc.txt", "FedDSA")
analyze("/tmp/fdse_acc.txt", "FDSE")
analyze("/tmp/fedbn_acc.txt", "FedBN")
