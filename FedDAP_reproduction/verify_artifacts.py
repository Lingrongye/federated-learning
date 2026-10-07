"""Post-run acceptance: check ALL rounds, real gradient use, and heavy snapshots."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_digits import backbone, sha256


def verify(directory):
    path = Path(directory)
    meta = json.loads((path / "meta.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "data_manifest.json").read_text())
    assert summary["status"] == "completed"
    assert summary["paper_result_reproduced"] is False
    expected = meta["args"]["rounds"]
    records = [json.loads(line) for line in (path / "proto_logs.jsonl").read_text().splitlines()]
    assert len(records) == expected
    assert len(list(path.glob("round_*.npz"))) == expected
    for domain in meta["domains"]:
        sets = [set(item["indices"]) for item in manifest["clients"] if item["domain"] == domain]
        assert len(set.union(*sets)) == sum(map(len, sets)), "Client sample overlap"
    for round_id, record in enumerate(records, 1):
        with np.load(path / f"round_{round_id:03d}.npz", allow_pickle=False) as item:
            assert int(item["round"]) == round_id
            expected_acc = item["test_correct"] / item["test_count"] * 100
            assert np.allclose(expected_acc, item["domain_accuracy"])
            assert np.isclose(expected_acc.mean(), item["avg_accuracy"])
            counts = item["client_sample_counts"]
            assert np.allclose(counts / counts.sum(), item["fedavg_weights"])
            assert np.isfinite(item["prototype_values"]).all()
            assert np.isclose(sum(item["fedavg_weights"]), 1)
            assert len(item["prototype_classes"]) == record["global_prototype_count"]
        for client in record["clients"]:
            assert np.isfinite([client[key] for key in ("ce", "dpa", "cpcl")]).all()
            if round_id == 1:
                assert client["dpa"] == 0 and client["cpcl"] == 0
            else:
                assert client["valid_dpa"] > 0 and client["valid_cpcl"] > 0
                assert client["dpa_feature_grad"] > 0 and client["cpcl_feature_grad"] > 0
        for attention in record["attention"].values():
            assert np.isclose(sum(attention["weights"]), 1)
    final = path / f"final_R{expected:03d}.npz"
    snapshots = list(path.glob("best_R*.npz")) + [final]
    assert len(snapshots) >= 2
    net, _ = backbone()
    for snapshot in snapshots:
        with np.load(snapshot, allow_pickle=False) as item:
            state = {key.split("model::", 1)[1]: torch.from_numpy(item[key])
                     for key in item.files if key.startswith("model::")}
            net.load_state_dict(state, strict=True)
            assert item["features"].shape[0] == len(item["labels"])
            with torch.no_grad():
                prediction = net.classifier(torch.from_numpy(item["features"])).argmax(1).numpy()
            for index, domain in enumerate(meta["domains"]):
                mask = item["feature_domains"] == domain
                hits = int((prediction[mask] == item["labels"][mask]).sum())
                assert hits == int(item["test_correct"][index]), "Classifier/feature snapshot mismatch"
                assert int(mask.sum()) == int(item["test_count"][index])
    files = sorted(file for file in path.iterdir() if file.is_file())
    proof = {
        "acceptance": "PASS", "rounds": expected, "clients": len(manifest["clients"]),
        "test_samples": sum(item["count"] for item in manifest["test"]),
        "second_round_both_losses_and_gradients": True,
        "all_snapshot_model_and_feature_pairs_verified": True,
        "source_revision": summary["revision"],
        "files": {file.name: {"bytes": file.stat().st_size, "sha256": sha256(file)} for file in files},
    }
    print(json.dumps(proof, indent=2), flush=True)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    verify(parser.parse_args().directory)
