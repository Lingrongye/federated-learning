#!/usr/bin/env python3
"""
Collect experiment results from flgo's default output locations
into the corresponding experiments/ directory.

Usage:
    python collect_results.py --exp EXP-052 --task PACS_c4 --algorithm feddsa --seed 2
    python collect_results.py --exp EXP-052 --task PACS_c4 --algorithm feddsa  # all seeds

This copies:
    task/PACS_c4/record/*feddsa*S2*.json  →  experiments/.../EXP-052/results/
    task/PACS_c4/log/*feddsa*S2*.log      →  experiments/.../EXP-052/logs/
"""
import argparse
import glob
import os
import shutil
import json


def find_exp_dir(exp_id):
    """Find the experiment directory by EXP-XXX id."""
    base = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if d.startswith(exp_id):
                return os.path.join(root, d)
    return None


def collect(exp_id, task, algorithm, seed=None):
    exp_dir = find_exp_dir(exp_id)
    if exp_dir is None:
        print(f"ERROR: {exp_id} directory not found under experiments/")
        return

    record_dir = os.path.join('task', task, 'record')
    log_dir = os.path.join('task', task, 'log')
    out_results = os.path.join(exp_dir, 'results')
    out_logs = os.path.join(exp_dir, 'logs')
    os.makedirs(out_results, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)

    # Build pattern
    seed_pat = f"_S{seed}_" if seed else "_S*_"
    algo_pat = f"*{algorithm}*"

    # Collect JSON
    json_files = [f for f in glob.glob(os.path.join(record_dir, '*.json'))
                  if algorithm in os.path.basename(f) and (seed is None or f"_S{seed}_" in os.path.basename(f))]

    for f in json_files:
        dst = os.path.join(out_results, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
            print(f"  JSON: {os.path.basename(f)} → results/")
        else:
            print(f"  JSON: {os.path.basename(f)} (already exists)")

    # Collect LOG
    log_files = [f for f in glob.glob(os.path.join(log_dir, '*.log'))
                 if algorithm in os.path.basename(f) and (seed is None or f"_S{seed}_" in os.path.basename(f))]

    for f in log_files:
        dst = os.path.join(out_logs, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
            print(f"  LOG: {os.path.basename(f)} → logs/")
        else:
            print(f"  LOG: {os.path.basename(f)} (already exists)")

    # Summary
    total = len(json_files) + len(log_files)
    if total == 0:
        print(f"  WARNING: No files found for {algorithm} seed={seed} in {task}")
    else:
        print(f"  Collected {len(json_files)} json + {len(log_files)} log → {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help='Experiment ID (e.g. EXP-052)')
    parser.add_argument('--task', required=True, help='Task name (e.g. PACS_c4)')
    parser.add_argument('--algorithm', required=True, help='Algorithm name substring')
    parser.add_argument('--seed', type=int, default=None, help='Specific seed (optional)')
    args = parser.parse_args()
    collect(args.exp, args.task, args.algorithm, args.seed)
