#!/bin/bash
# 推断每个 ckpt 是 PACS (7 类) 还是 Office (10 类)
PY=/home/lry/conda/envs/pfllib/bin/python
for ckpt in best108_1776428164 best122_1776452601 best191_1776428245 best196_1776473922; do
    echo "=== $ckpt ==="
    $PY -c "
import torch
sd = torch.load('/home/lry/fl_checkpoints/feddsa_s2_R200_$ckpt/global_model.pt', map_location='cpu')
print(f'  head.weight shape: {sd[\"head.weight\"].shape}, num_classes={sd[\"head.weight\"].shape[0]}')
"
done
