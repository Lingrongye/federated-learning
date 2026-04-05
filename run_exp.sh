#!/bin/bash
# ============================================================
# run_exp.sh — 实验启动器
# 功能: 后台运行实验 → 跑完自动通知(WxPusher) → 自动git commit+push结果
#
# 用法:
#   bash run_exp.sh <实验目录> <GPU_ID> <python参数...>
#
# 示例:
#   bash run_exp.sh experiments/sanity/EXP-002_pacs_feddsa_bugfix_verify 1 \
#       -data PACS -m ResNet18 -algo FedDSA -pt \
#       -ncl 7 -nc 4 -gr 50 -ls 5 -lr 0.005 -lbs 32 \
#       -eg 5 -t 1 -lo 1.0 -lh 0.1 -ls2 1.0 -tau 0.1 \
#       -wr 10 -sdn 5 -sdd 0.95 -sbm 50
# ============================================================

set -euo pipefail

# -------- 配置 --------
PROJECT_DIR="/home/lry/code/federated-learning"
PYTHON="/home/lry/conda/envs/pfllib/bin/python"
MAIN_PY="${PROJECT_DIR}/PFLlib/system/main.py"

# WxPusher 通知配置（填入你的 token 和 uid）
WXPUSHER_APP_TOKEN="AT_v9VClDlgaiYFGMLX9Sp5D0TpqTS36oc8"
WXPUSHER_UID="UID_n0UJoFp2uqba9z4M4Jfaf6N30sIc"
# -------- 配置结束 --------

# 参数检查
if [ $# -lt 3 ]; then
    echo "用法: bash run_exp.sh <实验目录(相对项目根)> <GPU_ID> <python参数...>"
    echo "示例: bash run_exp.sh experiments/sanity/EXP-002 1 -data PACS -m ResNet18 -algo FedDSA ..."
    exit 1
fi

EXP_DIR_REL="$1"
GPU_ID="$2"
shift 2
PYTHON_ARGS="$@"

EXP_DIR="${PROJECT_DIR}/${EXP_DIR_REL}"
EXP_NAME=$(basename "$EXP_DIR_REL")

# 确保实验目录存在
mkdir -p "${EXP_DIR}/results"

# -------- 通知函数 --------
notify() {
    local title="$1"
    local content="$2"
    if [ -n "$WXPUSHER_APP_TOKEN" ] && [ -n "$WXPUSHER_UID" ]; then
        # 用python构造JSON，避免特殊字符破坏格式
        ${PYTHON} -c "
import json, sys, urllib.request
data = json.dumps({
    'appToken': '${WXPUSHER_APP_TOKEN}',
    'content': sys.stdin.read(),
    'summary': '''${title}'''[:100],
    'contentType': 1,
    'uids': ['${WXPUSHER_UID}']
}).encode('utf-8')
req = urllib.request.Request('https://wxpusher.zjiecode.com/api/send/message',
    data=data, headers={'Content-Type': 'application/json; charset=utf-8'})
urllib.request.urlopen(req, timeout=10)
" <<< "$content" 2>/dev/null || true
        echo "[notify] sent: ${title}"
    else
        echo "[notify] WxPusher not configured, skipping"
    fi
}

# -------- 提取结果 --------
extract_summary() {
    local log_file="${EXP_DIR}/terminal.log"
    if [ -f "$log_file" ]; then
        grep -E "\[RESULT\]" "$log_file" | head -10 || echo "No RESULT lines"
    else
        echo "No terminal.log"
    fi
}

extract_full_log() {
    local log_file="${EXP_DIR}/terminal.log"
    if [ -f "$log_file" ]; then
        # WxPusher限制4万字符，截取尾部保留最重要的信息
        tail -c 38000 "$log_file"
    else
        echo "No terminal.log"
    fi
}

# -------- 主流程 --------
echo "============================================"
echo "[run_exp] 实验: ${EXP_NAME}"
echo "[run_exp] GPU: ${GPU_ID}"
echo "[run_exp] 目录: ${EXP_DIR}"
echo "[run_exp] 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 运行实验
cd "${PROJECT_DIR}/PFLlib/system"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${MAIN_PY} \
    ${PYTHON_ARGS} \
    -did ${GPU_ID} \
    -edir "${EXP_DIR}"
EXIT_CODE=$?

FINISH_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "[run_exp] 结束时间: ${FINISH_TIME}"
echo "[run_exp] 退出码: ${EXIT_CODE}"

# 提取日志
FULL_LOG=$(extract_full_log)

# 发送通知（完整日志）
if [ $EXIT_CODE -eq 0 ]; then
    notify "${EXP_NAME} done" "${FULL_LOG}"
else
    notify "${EXP_NAME} FAILED (exit=${EXIT_CODE})" "${FULL_LOG}"
fi

# 自动 git commit + push 结果
cd "${PROJECT_DIR}"
if git diff --quiet HEAD -- "${EXP_DIR_REL}/" 2>/dev/null && \
   [ -z "$(git ls-files --others --exclude-standard "${EXP_DIR_REL}/")" ]; then
    echo "[run_exp] 无新文件变更，跳过git提交"
else
    echo "[run_exp] 提交实验结果到Git..."
    git add "${EXP_DIR_REL}/"
    git commit -m "结果: ${EXP_NAME} (exit=${EXIT_CODE})" || true
    git push origin main || echo "[run_exp] git push失败，需手动推送"
    echo "[run_exp] Git提交完成"
fi

echo "[run_exp] 全部完成"
