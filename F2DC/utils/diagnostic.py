"""
诊断数据 hook — 训练时只 dump 原始 binary, 不算任何衍生指标 (cold path 后处理).

设计原则:
- Hot path 0.1-0.5% wall overhead (只 dump 元数据 + 偶发 best/final heavy snapshot)
- 所有衍生指标 (t-SNE / silhouette / CKA / cos sim trajectory 等) 在 cold path script 算
- 同一份 dump 可以被反复 mine 出新指标 (不用重跑训练)

Dump 结构:
  diag_dir/
    round_001.npz          # 元数据 ~50 KB
    round_002.npz
    ...
    round_100.npz
    best_R042.npz          # 第 1 次 best ~5 MB (含 features + state_dict)
    best_R068.npz          # 第 2 次 best (gain ≥ 1pp 触发)
    ...
    final_R100.npz         # 最后 round
    meta.json              # 实验配置 + 总结

用法:
  在 main_run.py 加 --dump_diag /path/to/dir 即启用.
  不传则 dump_round_metadata 跟 dump_heavy_snapshot 都 noop.
"""
import os
import json
import numpy as np
import torch


# ============================================================
# Light dump: 每 round 元数据 (50 KB, ~10 ms)
# ============================================================

def dump_round_metadata(model, round_idx, eval_results, all_dataset_names, args):
    """每 round 调用一次, 只 dump 元数据 不算任何衍生指标.

    内容:
    - sample_shares + daa_freqs (DaA dispatch 对照)
    - global_proto + per-client local_protos (cos sim trajectory)
    - per-layer L2 drift (PG-DFC vs PG-DFC+DaA saturate 检测)
    - per-domain acc (主表)
    """
    diag_dir = getattr(args, 'dump_diag', None)
    if not diag_dir:
        return
    os.makedirs(diag_dir, exist_ok=True)

    online_clients = getattr(model, 'online_clients', list(range(args.parti_num)))
    K = len(online_clients)

    # === sample shares (FedAvg baseline 对照) ===
    sample_shares = np.zeros(K, dtype=np.float32)
    if hasattr(model, 'trainloaders') and model.trainloaders:
        try:
            lens = [model.trainloaders[i].sampler.indices.size for i in online_clients]
            total = float(sum(lens))
            sample_shares = np.array([n / total for n in lens], dtype=np.float32)
        except Exception:
            pass

    # === DaA freq (如果用了 use_daa, 在 _compute_daa_freq 后已暴露) ===
    daa_freqs = getattr(model, '_last_daa_freq', None)
    if daa_freqs is not None:
        daa_freqs = np.asarray(daa_freqs, dtype=np.float32)
    else:
        daa_freqs = np.zeros(K, dtype=np.float32)  # 没用 DaA, 全零作 sentinel

    # === global proto (server 端 cross-client class proto) ===
    global_proto_arr = None
    proto_obj = getattr(model, 'global_protos', None)
    if proto_obj:
        # 不同 model 可能 dict 或 tensor — 都 normalize 成 (C, D) array fp16
        try:
            if isinstance(proto_obj, dict):
                # FedProto: {label: tensor(D,)} or {label: [tensor]}
                C = args.num_classes
                D = None
                slots = []
                for c in range(C):
                    val = proto_obj.get(c, None)
                    if val is None:
                        slots.append(None)
                    else:
                        v = val[0] if isinstance(val, list) else val
                        slots.append(v.detach().cpu().numpy())
                        D = D or slots[-1].shape[-1]
                if D is not None:
                    matrix = np.zeros((C, D), dtype=np.float16)
                    for c, s in enumerate(slots):
                        if s is not None:
                            matrix[c] = s.astype(np.float16)
                    global_proto_arr = matrix
            elif torch.is_tensor(proto_obj):
                global_proto_arr = proto_obj.detach().cpu().numpy().astype(np.float16)
        except Exception as e:
            print(f"[diag] global_proto dump failed: {e}")

    # === local protos (per-client per-class) ===
    # 兼容 3 种 model 形式:
    # - PG-DFC: list[parti_num] of tensor (C, D) — index 直接索引
    # - FedProto: dict[client_id] -> {label: tensor} (list-wrapped possible)
    # - 其它: None
    local_protos_arr = None
    local_protos_obj = getattr(model, 'local_protos', None)
    if local_protos_obj is not None:
        try:
            C = args.num_classes
            D_local = None
            stack = []
            for ki, i in enumerate(online_clients):
                # 处理 list 形式 (PG-DFC)
                if isinstance(local_protos_obj, list):
                    if i < len(local_protos_obj) and local_protos_obj[i] is not None:
                        t = local_protos_obj[i]
                        if torch.is_tensor(t):
                            arr = t.detach().cpu().numpy()  # (C, D)
                            D_local = D_local or arr.shape[-1]
                            stack.append(arr.astype(np.float16))
                            continue
                    stack.append(None)
                    continue
                # 处理 dict 形式 (FedProto-like)
                if isinstance(local_protos_obj, dict):
                    client_proto = local_protos_obj.get(i, None)
                    if isinstance(client_proto, dict):
                        slots = []
                        for c in range(C):
                            val = client_proto.get(c, None)
                            if val is None:
                                slots.append(None)
                            else:
                                v = val[0] if isinstance(val, list) else val
                                if torch.is_tensor(v):
                                    slots.append(v.detach().cpu().numpy())
                                    D_local = D_local or slots[-1].shape[-1]
                                else:
                                    slots.append(None)
                        if D_local:
                            m = np.zeros((C, D_local), dtype=np.float16)
                            for c, s in enumerate(slots):
                                if s is not None:
                                    m[c] = s.astype(np.float16)
                            stack.append(m)
                        else:
                            stack.append(None)
                    else:
                        stack.append(None)
                else:
                    stack.append(None)

            if any(s is not None for s in stack):
                D_local = D_local or 512
                arr_full = np.zeros((K, C, D_local), dtype=np.float16)
                for ki, s in enumerate(stack):
                    if s is not None and s.shape == (C, D_local):
                        arr_full[ki] = s
                local_protos_arr = arr_full
        except Exception as e:
            print(f"[diag] local_protos dump failed: {e}")

    # === per-layer L2 drift (client_i vs global) — 只 dump 几个关键 layer 名 ===
    layer_l2 = {}
    try:
        if hasattr(model, 'global_net') and model.global_net is not None:
            global_sd = model.global_net.state_dict()
            critical_layers = []
            # auto detect: conv1 / layer1.0/layer4.0 / linear / classifier
            for k in global_sd.keys():
                if any(p in k for p in ['conv1.weight', 'layer1.0.conv1', 'layer4.0.conv1',
                                        'classifier.weight', 'linear.weight']):
                    critical_layers.append(k)
            if not critical_layers:
                # fallback first 3 weight layers
                critical_layers = [k for k in global_sd if k.endswith('.weight')][:3]
            for ki, client_id in enumerate(online_clients):
                client_sd = model.nets_list[client_id].state_dict()
                client_drift = {}
                for layer in critical_layers:
                    if layer in client_sd and layer in global_sd:
                        diff = (client_sd[layer].float() - global_sd[layer].float())
                        client_drift[layer] = float(diff.norm().item())
                layer_l2[client_id] = client_drift
    except Exception as e:
        print(f"[diag] layer_l2 dump failed: {e}")

    # === gradient L2 (= ||client - global|| 总和) ===
    grad_l2 = np.zeros(K, dtype=np.float32)
    try:
        if layer_l2:
            for ki, client_id in enumerate(online_clients):
                if client_id in layer_l2:
                    grad_l2[ki] = sum(v ** 2 for v in layer_l2[client_id].values()) ** 0.5
    except Exception:
        pass

    # === domain per client (跟我们 setup 一致, 用于 cold path 按 domain 分组) ===
    selected_domain_list = getattr(model, '_selected_domain_list', None)
    if selected_domain_list is not None:
        domain_per_client = np.array([str(d) for d in selected_domain_list], dtype='<U16')
    else:
        domain_per_client = np.array(['unknown'] * args.parti_num, dtype='<U16')

    # === per-domain test acc + per-class confusion (eval_results 里的) ===
    per_domain_acc = np.array(eval_results, dtype=np.float32)

    # === save ===
    save_kwargs = dict(
        round=int(round_idx),
        sample_shares=sample_shares,
        daa_freqs=daa_freqs,
        online_clients=np.array(online_clients, dtype=np.int32),
        domain_per_client=domain_per_client,
        per_domain_acc=per_domain_acc,
        all_dataset_names=np.array(all_dataset_names, dtype='<U16'),
        grad_l2=grad_l2,
    )
    if global_proto_arr is not None:
        save_kwargs['global_proto'] = global_proto_arr
    if local_protos_arr is not None:
        save_kwargs['local_protos'] = local_protos_arr
    # layer_l2 是 nested dict, 用 pickle (np.savez 不支持 dict)
    save_kwargs['layer_l2_pickle'] = np.array([json.dumps(layer_l2)], dtype=object)

    np.savez_compressed(
        os.path.join(diag_dir, f"round_{round_idx:03d}.npz"),
        **save_kwargs,
    )


# ============================================================
# Heavy dump: best + final 触发 (5 MB, ~5 sec)
# ============================================================

class _HeavySnapshotState:
    """跟 args 一起 持续状态: 跟踪 best 触发记录."""
    def __init__(self):
        self.last_dumped_round = -1
        self.last_dumped_best_acc = -1.0
        self.dumps_taken = []


def should_dump_heavy_best(round_idx, current_best_acc, args, state):
    """判断是否在当前 round dump heavy snapshot."""
    diag_dir = getattr(args, 'dump_diag', None)
    if not diag_dir:
        return False
    warmup = int(getattr(args, 'dump_warmup', 30))
    min_gain = float(getattr(args, 'dump_min_gain', 1.0))
    min_interval = int(getattr(args, 'dump_min_interval', 5))
    if round_idx < warmup:
        return False
    if current_best_acc - state.last_dumped_best_acc < min_gain:
        return False
    if round_idx - state.last_dumped_round < min_interval:
        return False
    return True


def dump_heavy_snapshot(model, test_loaders, prefix, round_idx, current_acc, args, state):
    """dump 全 test set features + state_dict + per-class confusion.

    被 best round 跟 final round 调用. 单次 ~5 MB ~5 sec.
    """
    diag_dir = getattr(args, 'dump_diag', None)
    if not diag_dir:
        return
    os.makedirs(diag_dir, exist_ok=True)

    device = model.device
    model.global_net.eval()
    status_was_training = model.global_net.training

    all_features = {}
    all_labels = {}
    all_preds = {}
    all_logits = {}
    confusion = {}

    if model.args.dataset == "fl_officecaltech":
        domain_names = ["caltech", "amazon", "webcam", "dslr"]
    elif model.args.dataset == "fl_digits":
        domain_names = ["mnist", "usps", "svhn", "syn"]
    elif model.args.dataset == "fl_pacs":
        domain_names = ["photo", "art", "cartoon", "sketch"]
    else:
        domain_names = [f"d{i}" for i in range(len(test_loaders))]

    NC = int(args.num_classes)
    use_tuple = model.NAME in ("f2dc", "f2dc_pg", "f2dc_pgv33")

    with torch.no_grad():
        for di, dl in enumerate(test_loaders):
            domain_name = domain_names[di] if di < len(domain_names) else f"d{di}"
            feats_list, labels_list, preds_list, logits_list = [], [], [], []
            for x, y in dl:
                x = x.to(device)
                y = y.to(device)
                # extract features + logits — 兼容 3 类 model:
                # 1. PG-DFC / F2DC ResNet_PG / ResNet_DC: forward 返回 7-tuple
                #    (out, feat, r_outputs, nr_outputs, rec_outputs, ro_flat, re_flat)
                # 2. 标准 ResNet (FedAvg/FedBN/FedProx/FedProto/MOON): 有 .features() 跟 .classifier()
                # 3. FDSE: ResNet_FDSE 有 .features() 但 forward 返回单 tensor
                if use_tuple:
                    out_tuple = model.global_net(x, is_eval=True) if 'is_eval' in model.global_net.forward.__code__.co_varnames else model.global_net(x)
                    if isinstance(out_tuple, tuple) and len(out_tuple) >= 2:
                        logit = out_tuple[0]
                        feat = out_tuple[1]
                    else:
                        logit = out_tuple
                        # fallback: 用 .features() 如果有
                        feat = model.global_net.features(x) if hasattr(model.global_net, 'features') else logit
                elif hasattr(model.global_net, 'features') and hasattr(model.global_net, 'classifier'):
                    feat = model.global_net.features(x)
                    logit = model.global_net.classifier(feat)
                elif hasattr(model.global_net, 'features'):
                    feat = model.global_net.features(x)
                    logit = model.global_net(x)
                    if isinstance(logit, tuple):
                        logit = logit[0]
                else:
                    logit = model.global_net(x)
                    if isinstance(logit, tuple):
                        feat = logit[1] if len(logit) >= 2 else logit[0]
                        logit = logit[0]
                    else:
                        feat = logit  # fallback, won't have penultimate features
                pred = logit.argmax(dim=-1)

                feats_list.append(feat.detach().cpu().numpy().astype(np.float16))
                labels_list.append(y.detach().cpu().numpy().astype(np.int32))
                preds_list.append(pred.detach().cpu().numpy().astype(np.int32))
                logits_list.append(logit.detach().cpu().numpy().astype(np.float16))

            f = np.concatenate(feats_list)
            l = np.concatenate(labels_list)
            p = np.concatenate(preds_list)
            lg = np.concatenate(logits_list)
            all_features[domain_name] = f
            all_labels[domain_name] = l
            all_preds[domain_name] = p
            all_logits[domain_name] = lg

            cm = np.zeros((NC, NC), dtype=np.int32)
            for true, pred in zip(l, p):
                if 0 <= int(true) < NC and 0 <= int(pred) < NC:
                    cm[int(true)][int(pred)] += 1
            confusion[domain_name] = cm

    if status_was_training:
        model.global_net.train()

    # === state_dict (fp16 省空间) ===
    state_dict_fp16 = {}
    for k, v in model.global_net.state_dict().items():
        try:
            state_dict_fp16[k] = v.detach().cpu().to(torch.float16).numpy()
        except Exception:
            try:
                state_dict_fp16[k] = v.detach().cpu().numpy()
            except Exception:
                pass

    save_path = os.path.join(diag_dir, f"{prefix}.npz")
    np.savez_compressed(
        save_path,
        round=int(round_idx),
        current_acc=float(current_acc),
        # features (dict 用 object array 保存)
        features=np.array([all_features], dtype=object),
        labels=np.array([all_labels], dtype=object),
        preds=np.array([all_preds], dtype=object),
        logits=np.array([all_logits], dtype=object),
        confusion=np.array([confusion], dtype=object),
        state_dict=np.array([state_dict_fp16], dtype=object),
        domain_names=np.array(domain_names, dtype='<U16'),
    )

    state.last_dumped_round = round_idx
    state.last_dumped_best_acc = current_acc
    state.dumps_taken.append((prefix, round_idx, current_acc))
    print(f"[diag] heavy snapshot dumped: {save_path} (acc={current_acc:.3f})")


def init_diag_state(args):
    """call once at start of train()."""
    diag_dir = getattr(args, 'dump_diag', None)
    if diag_dir:
        os.makedirs(diag_dir, exist_ok=True)
    return _HeavySnapshotState()


def dump_meta(args, state, total_rounds, final_acc):
    """call once at end of train()."""
    diag_dir = getattr(args, 'dump_diag', None)
    if not diag_dir:
        return
    meta = {
        'model': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'parti_num': args.parti_num,
        'num_classes': args.num_classes,
        'communication_epoch': args.communication_epoch,
        'use_daa': bool(getattr(args, 'use_daa', False)),
        'agg_a': float(getattr(args, 'agg_a', 1.0)),
        'agg_b': float(getattr(args, 'agg_b', 0.4)),
        'final_acc': float(final_acc),
        'heavy_dumps': state.dumps_taken,
    }
    with open(os.path.join(diag_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
