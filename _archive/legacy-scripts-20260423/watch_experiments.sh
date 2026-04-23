#!/bin/bash
# 监控所有run_single.py进程，结束时发WxPusher通知
PY=/root/miniconda3/bin/python
APP_TOKEN="AT_v9VClDlgaiYFGMLX9Sp5D0TpqTS36oc8"
UID_WX="UID_n0UJoFp2uqba9z4M4Jfaf6N30sIc"
LOG_DIR=/root/autodl-tmp/federated-learning/FDSE_CVPR25/task/PACS_c4/log

notify() {
    local title="$1"
    local content="$2"
    $PY -c "
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
    echo "[$(date '+%H:%M:%S')] notified: $title"
}

# 记录初始PID和对应算法名
declare -A PIDS
for pid in $(pgrep -f run_single.py); do
    algo=$(ps -p $pid -o args= 2>/dev/null | grep -oP '(?<=--algorithm )\S+' || echo unknown)
    config=$(ps -p $pid -o args= 2>/dev/null | grep -oP '(?<=--config )\S+' || echo "")
    PIDS[$pid]="$algo ($config)"
done

echo "[$(date '+%H:%M:%S')] Watching ${#PIDS[@]} experiments:"
for pid in "${!PIDS[@]}"; do
    echo "  PID $pid: ${PIDS[$pid]}"
done

# 轮询检查
while [ ${#PIDS[@]} -gt 0 ]; do
    for pid in "${!PIDS[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            name="${PIDS[$pid]}"
            echo "[$(date '+%H:%M:%S')] FINISHED: $name (PID $pid)"

            # 找对应的最新日志
            algo=$(echo "$name" | cut -d' ' -f1)
            latest_log=$(ls -t $LOG_DIR/*${algo}* 2>/dev/null | head -1)
            if [ -n "$latest_log" ]; then
                log_content=$(tail -c 35000 "$latest_log")
            else
                log_content="No log found for $algo"
            fi

            notify "$algo finished" "$log_content"
            unset PIDS[$pid]
        fi
    done
    sleep 30
done

echo "[$(date '+%H:%M:%S')] All experiments finished!"
notify "All experiments done" "All ${#PIDS[@]} experiments have completed on seetacloud."
