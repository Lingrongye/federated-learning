#!/bin/bash
# 在seetacloud服务器上并行跑3个基线
set -euo pipefail

PROJECT=/root/autodl-tmp/federated-learning
PYTHON=/root/miniconda3/bin/python
MAIN=$PROJECT/PFLlib/system/main.py
COMMON="-data PACS -m ResNet18 -pt -ncl 7 -nc 4 -gr 50 -ls 5 -lr 0.005 -lbs 32 -eg 5 -t 1"

# WxPusher
APP_TOKEN="AT_v9VClDlgaiYFGMLX9Sp5D0TpqTS36oc8"
UID_WX="UID_n0UJoFp2uqba9z4M4Jfaf6N30sIc"

notify() {
    local title="$1"
    local content="$2"
    $PYTHON -c "
import json, sys, urllib.request
content = sys.stdin.read()
data = json.dumps({
    'appToken': '$APP_TOKEN',
    'content': content,
    'summary': '''$title'''[:100],
    'contentType': 1,
    'uids': ['$UID_WX']
}).encode('utf-8')
req = urllib.request.Request('https://wxpusher.zjiecode.com/api/send/message',
    data=data, headers={'Content-Type': 'application/json; charset=utf-8'})
urllib.request.urlopen(req, timeout=10)
" <<< "$content" 2>/dev/null || true
    echo "[notify] $title"
}

run_one() {
    local name=$1
    local edir=$2
    shift 2
    local args="$@"

    mkdir -p "$edir/results"
    echo "[$(date '+%H:%M:%S')] START $name"

    cd $PROJECT/PFLlib/system
    CUDA_VISIBLE_DEVICES=0 $PYTHON main.py $COMMON $args -did 0 -edir "$edir" 2>&1 || true
    local exit_code=${PIPESTATUS[0]:-$?}

    echo "[$(date '+%H:%M:%S')] DONE $name (exit=$exit_code)"

    # notify
    local log_content=$(tail -c 38000 "$edir/terminal.log" 2>/dev/null || echo "no log")
    if [ $exit_code -eq 0 ]; then
        notify "$name done" "$log_content"
    else
        notify "$name FAILED (exit=$exit_code)" "$log_content"
    fi

    # git commit
    cd $PROJECT
    git add "$edir/" 2>/dev/null || true
    git commit -m "result: $name (exit=$exit_code)" 2>/dev/null || true
    git push origin main 2>/dev/null || true

    return $exit_code
}

echo "===== Launching 3 baselines in parallel on GPU 0 ====="
echo "Time: $(date)"

run_one "EXP-003 FedAvg" "$PROJECT/experiments/baselines/EXP-003_pacs_fedavg" -algo FedAvg &
PID1=$!

run_one "EXP-004 FedBN" "$PROJECT/experiments/baselines/EXP-004_pacs_fedbn" -algo FedBN &
PID2=$!

run_one "EXP-005 FedProto" "$PROJECT/experiments/baselines/EXP-005_pacs_fedproto" -algo FedProto -lam 1.0 &
PID3=$!

echo "PIDs: FedAvg=$PID1 FedBN=$PID2 FedProto=$PID3"

wait $PID1; R1=$?
wait $PID2; R2=$?
wait $PID3; R3=$?

echo "===== All done: FedAvg=$R1 FedBN=$R2 FedProto=$R3 ====="

# final summary notify
SUMMARY="3 baselines completed:
- EXP-003 FedAvg: exit=$R1
- EXP-004 FedBN: exit=$R2
- EXP-005 FedProto: exit=$R3

$(grep -h RESULT $PROJECT/experiments/baselines/*/terminal.log 2>/dev/null || echo 'no RESULT lines')"

notify "All 3 baselines done" "$SUMMARY"
