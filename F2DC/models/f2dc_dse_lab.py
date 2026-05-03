"""
F2DC + DSE_Rescue3 + LAB v4.2 (Loss-Aware Boost) Aggregation
==================================================================

继承 F2DCDSE (= F2DC base + DSE_Rescue3 + CCC + Mag), 唯一改动:
  把 server 端 sample-weighted FedAvg 替换为 LAB v4.2:
  client 持本地 held-out val (从 partition unused 反推) → 上传 (loss_sum, val_n)
  → server sample-weighted 聚合 val_loss → ReLU(loss - mean) underfit boost
  → λ-mix + bounded simplex projection → 域内等权 → 聚合权重 freq.

设计原则:
1. **完整保留 DSE 所有逻辑** (rho_t / lambda_cc_t / proto3 EMA / CCC loss / Mag guard /
   L_orth / proto3_*_diag / mag_*_diag / ccc_improvement 等)
2. **完整保留 LAB 所有诊断** (lab_ratio_<dom>, lab_boost_<dom>, lab_clip_at_max_<dom>,
   val_class_counts, lab_used_this_round, lab_cli_freq_<k> 等)
3. **聚合范围**:
   - **backbone state_dict**: 走 LAB freq (替代 FedAvg sample-weighted)
   - **DSE proto3 EMA**: 仍走 sample-mean per class (跟 F2DCDSE 一致, NOT LAB 化)
     原因: proto3 是 class prototype, 服务于 CCC 监督方向, 不是 backbone 参数;
     按 class 平均更稳, 没必要 LAB 加权 (codex review 确认这是设计选择)
4. 不改原有 F2DCDSE / F2DCPgLab 文件, 复制 LAB method 进来 (用户要求)

接入:
    --model f2dc_dse_lab \
    --use_daa false \                        # 跟 LAB 互斥
    --dse_rho_max 0.3 \                      # DSE 修正强度 (PACS winner 0.3)
    --dse_lambda_cc 0.1 \
    --dse_lambda_mag 0.01 \
    --lab_lambda 0.15 \                      # LAB 总加成预算
    --lab_ratio_min 0.80 \
    --lab_ratio_max 2.00 \
    --lab_projection_mode standard           # PACS 默认; Office 可用 office_small_protect

PROPOSAL: experiments/ablation/EXP-144_lab_v4_1/PROPOSAL.md (LAB)
EXP-143 NOTE: PACS rho=0.3 mean best 73.40 +2.38 (DSE winner)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils.args import *
from models.f2dc_dse import F2DCDSE
from models.f2dc import get_pred
from models.utils.lab_aggregation import LabState
from datasets.utils.lab_partition import setup_lab_val_loaders, evaluate_val_per_client


class F2DCDseLab(F2DCDSE):
    """F2DC + DSE_Rescue3 + LAB aggregation.

    继承 F2DCDSE: 复用所有 DSE 训练 / proto3 EMA / CCC / Mag / L_orth / round diag.
    覆盖: aggregate_nets (LAB freq 替代 FedAvg sample-weighted), 在 loc_update 末尾加
          val_loss eval + LAB diag merge.
    """
    NAME = 'f2dc_dse_lab'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)

        # ===== LAB 超参 (跟 F2DCPgLab 一致, PROPOSAL §10) =====
        self.lab_lambda = float(getattr(args, 'lab_lambda', 0.15))
        self.lab_ratio_min = float(getattr(args, 'lab_ratio_min', 0.80))
        self.lab_ratio_max = float(getattr(args, 'lab_ratio_max', 2.00))
        self.lab_projection_mode = str(getattr(args, 'lab_projection_mode', 'standard'))
        self.lab_small_share_threshold = float(getattr(args, 'lab_small_share_threshold', 0.125))
        self.lab_small_ratio_min = float(getattr(args, 'lab_small_ratio_min', 1.25))
        self.lab_small_ratio_max = float(getattr(args, 'lab_small_ratio_max', 4.00))
        self.lab_ema_alpha = float(getattr(args, 'lab_ema_alpha', 0.30))
        self.lab_val_size_per_dom = int(getattr(args, 'lab_val_size_per_dom', 50))
        self.lab_val_per_class = int(getattr(args, 'lab_val_per_class', 5))
        self.lab_val_seed = int(getattr(args, 'lab_val_seed', 42))
        self.lab_window_size = int(getattr(args, 'lab_window_size', 20))
        self.lab_waste_roi_threshold = float(getattr(args, 'lab_waste_roi_threshold', 0.5))
        self.lab_print_diag = bool(getattr(args, 'lab_print_diag', True))
        self.lab_warn_interval = int(getattr(args, 'lab_warn_interval', 10))

        # ===== LAB 跨 round 状态 =====
        self.lab_state = LabState(
            lam=self.lab_lambda,
            ratio_min=self.lab_ratio_min,
            ratio_max=self.lab_ratio_max,
            ema_alpha=self.lab_ema_alpha,
            window_size=self.lab_window_size,
            waste_roi_threshold=self.lab_waste_roi_threshold,
            projection_mode=self.lab_projection_mode,
            small_share_threshold=self.lab_small_share_threshold,
            small_ratio_min=self.lab_small_ratio_min,
            small_ratio_max=self.lab_small_ratio_max,
        )

        # val_loaders 在 first round 才 setup (trainloaders / _selected_domain_list
        # 在 utils/training.py setup 阶段才挂到 model 上, __init__ 时还没有)
        self.val_loaders = None
        self.val_meta = None
        self._lab_setup_done = False

        # 上一 round 计算出的 LAB 权重 (用于本 round 聚合)
        # round 1 时 None → 走 FedAvg fallback
        self._next_round_lab_freq = None
        self._next_round_lab_result = None

        # 兼容 use_daa 互斥 (paper-grade 严格)
        if getattr(args, 'use_daa', False):
            print("[LAB WARNING] --use_daa=True is incompatible with f2dc_dse_lab; "
                  "DaA is auto-disabled, using LAB instead.")
            self.args.use_daa = False

    # ============================================================
    # Per-round test acc hook (utils/training.py 在 evaluate 之后调用)
    # ============================================================
    def lab_record_test_acc(self, round_idx: int, accs, all_dataset_names) -> None:
        """utils/training.py global_evaluate 完后调一次, 把 per-domain test acc 喂给 LabState.
        Patch proto_logs[-1].lab_delta_acc_* 反映本 round 真实 (acc[R] - acc[R-1]).
        Trigger waste warning 用 evaluate 后的真实 acc.
        """
        if not hasattr(self, "lab_state") or self.lab_state is None:
            return
        try:
            acc_dict = {str(d): float(a) for d, a in zip(all_dataset_names, accs)}
            self.lab_state.update_test_acc(round_idx, acc_dict)

            # Patch proto_logs[-1].lab_delta_acc_*
            if hasattr(self, "proto_logs") and self.proto_logs:
                latest = self.proto_logs[-1]
                if isinstance(latest, dict):
                    for d in acc_dict:
                        ah = self.lab_state.acc_history.get(d, [])
                        if len(ah) >= 2:
                            latest[f"lab_delta_acc_{d}"] = float(ah[-1] - ah[-2])
                        else:
                            latest[f"lab_delta_acc_{d}"] = 0.0

            # Live waste warning (evaluate 后用真实 acc)
            if (round_idx > 0
                    and getattr(self, "lab_warn_interval", 0) > 0
                    and round_idx % self.lab_warn_interval == 0):
                try:
                    self._print_waste_warnings(round_idx_override=round_idx)
                except Exception as _w_err:
                    print(f"[LAB waste warn ERR] {_w_err}")
        except Exception as e:
            print(f"[LAB lab_record_test_acc ERR] {e}")

    # ============================================================
    # First-round val setup (从 trainloaders 反推 unused → val pool)
    # ============================================================
    def _ensure_lab_setup(self):
        """在第一次 loc_update 调用时跑."""
        if self._lab_setup_done:
            return
        if not hasattr(self, 'trainloaders') or self.trainloaders is None:
            print("[LAB ERROR] trainloaders not yet set, val pool setup deferred.")
            return
        if not hasattr(self, '_selected_domain_list') or self._selected_domain_list is None:
            print("[LAB ERROR] _selected_domain_list not set; val pool setup deferred.")
            return

        train_dataset_list = [dl.dataset for dl in self.trainloaders]
        try:
            val_loaders, val_meta = setup_lab_val_loaders(
                trainloaders=self.trainloaders,
                train_dataset_list=train_dataset_list,
                selected_domain_list=self._selected_domain_list,
                args=self.args,
                val_size_per_dom=self.lab_val_size_per_dom,
                val_per_class=self.lab_val_per_class,
                val_seed=self.lab_val_seed,
            )
        except Exception as e:
            print(f"[LAB ERROR] setup_lab_val_loaders failed: {e}")
            self.val_loaders = [None] * self.args.parti_num
            self.val_meta = {"val_n_per_cli": {}, "val_n_per_dom": {}, "val_class_counts": {}}
            self._lab_setup_done = True
            return

        self.val_loaders = val_loaders
        self.val_meta = val_meta
        self._lab_setup_done = True

        n_total = sum(val_meta["val_n_per_cli"].values())
        n_active_cli = len(val_meta["val_n_per_cli"])
        print(f"[LAB setup] val partition: {n_total} samples across {n_active_cli} clients, "
              f"per-dom = {val_meta['val_n_per_dom']}")
        for d, cls in val_meta.get("val_class_counts", {}).items():
            print(f"           {d} class counts: {dict(sorted(cls.items()))}")

    # ============================================================
    # 主 round 循环 (override F2DCDSE.loc_update — DSE + LAB 合并)
    # ============================================================
    def loc_update(self, priloader_list):
        # First-round val setup
        self._ensure_lab_setup()

        # === DSE Step 1: 算当前 round 的 rho_t / lambda_cc_t (跟 F2DCDSE 一致) ===
        t = self.epoch_index
        rho_t = self._compute_ramp_value(self.dse_rho_max,
                                          self.dse_rho_warmup, self.dse_rho_ramp, t)
        lambda_cc_t = self._compute_ramp_value(self.dse_lambda_cc,
                                                self.dse_cc_warmup, self.dse_cc_ramp, t)
        self._cur_rho_t = rho_t
        self._cur_lambda_cc_t = lambda_cc_t

        # === DSE Step 2: 同步 rho_t + global_proto3_unit + reset diag ===
        all_nets = list(self.nets_list)
        if hasattr(self, 'global_net') and self.global_net is not None:
            all_nets.append(self.global_net)
        for net in all_nets:
            if hasattr(net, 'set_rho_t'):
                net.set_rho_t(rho_t)
            if hasattr(net, 'set_global_proto3_unit') and self.global_proto3_unit is not None:
                net.set_global_proto3_unit(self.global_proto3_unit)
            if hasattr(net, 'reset_dse_diag'):
                net.reset_dse_diag()

        # === DSE Step 3: 重置 round 级累积 ===
        N_CLASSES = self.nets_list[0].linear.out_features
        feat3_dim = self.nets_list[0].feat3_dim if hasattr(self.nets_list[0], 'feat3_dim') else 256
        self._round_local_proto3_sum = torch.zeros(N_CLASSES, feat3_dim, device=self.device)
        self._round_local_proto3_count = torch.zeros(N_CLASSES, device=self.device)
        self._round_cc_loss_sum = 0.0
        self._round_mag_loss_sum = 0.0
        self._round_mag_exceed_samples = 0
        self._round_mag_eval_samples = 0
        self._round_total_batches = 0
        self._round_raw_to_target_cos = []
        self._round_rescued_to_target_cos = []
        self._round_mag_ratio_p95 = []
        self._round_mag_ratio_max = []
        self._round_proto3_ema_delta_norm = 0.0
        self._round_orth_loss_sum = 0.0
        self._round_orth_cos_abs_sum = 0.0
        self._round_orth_eval_samples = 0

        # === DSE Step 4: 标准 F2DC loc_update (call _train_net_dse, 继承的) ===
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num,
                                                    replace=False).tolist()
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []
        for i in online_clients:
            c_loss, c_samples = self._train_net_dse(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss
            self.num_samples.append(c_samples)

        # === LAB-replaced backbone 聚合 ===
        # aggregate_nets 会优先用 self._next_round_lab_freq;
        # round 1 (None) 时 fallback FedAvg sample-weighted (跟 vanilla 一致).
        self.aggregate_nets(None)

        # === DSE Step 6: server 端聚合 raw_proto3 + EMA + L2-norm + 同步 (继承的) ===
        self._aggregate_proto3()

        # === LAB Step 6 (codex guardrail #2): 聚合后才测 val_loss ===
        # 给下一 round LAB(r+1) 用.
        val_loss_per_dom = None
        val_loss_sum_per_cli = None
        val_n_per_cli = None
        if self._lab_setup_done and self.val_loaders is not None:
            try:
                eval_result = evaluate_val_per_client(
                    global_net=self.global_net,
                    val_loaders=self.val_loaders,
                    device=self.device,
                    selected_domain_list=self._selected_domain_list,
                )
                val_loss_per_dom = eval_result["val_loss_per_dom"]
                val_loss_sum_per_cli = eval_result["val_loss_sum_per_cli"]
                val_n_per_cli = eval_result["val_n_per_cli"]
                self.lab_state.update_val_loss(self.epoch_index, val_loss_per_dom)
            except Exception as e:
                print(f"[LAB val eval ERR] round {self.epoch_index}: {e}")

        # === DSE Step 7: round summary print + write to proto_logs[] (继承的) ===
        # _print_round_diag() 内部会 self.proto_logs.append(merged DSE diag dict)
        self._print_round_diag()

        # === LAB diag merge into proto_logs[-1] ===
        # F2DCDSE._print_round_diag 已经 append DSE diag 到 proto_logs, 我们 update LAB 字段进去
        lab_diag = self._compute_lab_round_diag(
            val_loss_per_dom=val_loss_per_dom,
            val_loss_sum_per_cli=val_loss_sum_per_cli,
            val_n_per_cli=val_n_per_cli,
        )
        if lab_diag and self.proto_logs:
            self.proto_logs[-1].update(lab_diag)

        # === LAB stdout 实时打印 (PROPOSAL §5.3) ===
        if self.lab_print_diag and self._next_round_lab_result is not None:
            self._print_lab_stdout(self.epoch_index, self._next_round_lab_result)

        # === 为下一 round 预算 LAB freq ===
        self._precompute_next_lab_freq()

        # 注意: F2DCDSE.loc_update 末尾 self.epoch_index += 1
        self.epoch_index += 1
        all_c_avg_loss = all_clients_loss / max(len(online_clients), 1)
        return round(all_c_avg_loss, 3)

    # ============================================================
    # LAB freq 预算 (给下一 round 用) — 跟 F2DCPgLab 完全一致
    # ============================================================
    def _precompute_next_lab_freq(self):
        if not hasattr(self, 'trainloaders') or self.trainloaders is None:
            self._next_round_lab_freq = None
            self._next_round_lab_result = None
            return

        sample_share_dom = self._compute_sample_share_dom(self.online_clients)
        if not sample_share_dom:
            self._next_round_lab_freq = None
            self._next_round_lab_result = None
            return

        result = self.lab_state.compute_lab(
            round_idx=self.epoch_index + 1,
            sample_share_dom=sample_share_dom,
        )
        self._next_round_lab_result = result
        self._next_round_lab_freq = result

    def _compute_sample_share_dom(self, online_clients):
        """从 online_clients 的 trainloader.sampler.indices 算 sample_share per domain."""
        if not hasattr(self, '_selected_domain_list') or self._selected_domain_list is None:
            return {}
        try:
            online_lens = []
            online_doms = []
            for k in online_clients:
                dl = self.trainloaders[k]
                n = dl.sampler.indices.size if hasattr(dl.sampler, "indices") else 0
                online_lens.append(int(n))
                online_doms.append(str(self._selected_domain_list[k]))
            total = float(sum(online_lens))
            if total <= 0:
                return {}
            sample_share_dom = {}
            for n, d in zip(online_lens, online_doms):
                sample_share_dom[d] = sample_share_dom.get(d, 0.0) + n / total
            return sample_share_dom
        except Exception as e:
            print(f"[LAB compute_sample_share_dom ERR] {e}")
            return {}

    def _domain_level_to_cli_freq(self, w_proj_dom, online_clients):
        """把 domain-level w_proj 转成 cli-level freq (域内等权)."""
        if not hasattr(self, '_selected_domain_list'):
            return None
        dom_to_cli_count = {}
        for k in online_clients:
            d = str(self._selected_domain_list[k])
            dom_to_cli_count[d] = dom_to_cli_count.get(d, 0) + 1

        freq = np.zeros(len(online_clients), dtype=np.float64)
        for i, k in enumerate(online_clients):
            d = str(self._selected_domain_list[k])
            n_in_dom = dom_to_cli_count.get(d, 1)
            w_d = float(w_proj_dom.get(d, 0.0))
            freq[i] = w_d / n_in_dom

        s = freq.sum()
        if s > 1e-9:
            freq = freq / s
        return freq

    # ============================================================
    # Override aggregate_nets — 用 LAB freq 替换 FedAvg
    # (跟 F2DCPgLab 完全一致: state_dict 全 key 聚合 + grad_l2/layer_l2 hook)
    # ============================================================
    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list
        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        # === diagnostic: 在 sync 前算 client 跟 global 的 drift ===
        try:
            old_global_keys = list(global_w.keys())
            critical_patterns = ['conv1.weight', 'layer1.0.conv1', 'layer4.0.conv1',
                                 'classifier.weight', 'linear.weight', 'cls.weight']
            actual_layers = [k for k in old_global_keys
                             if any(p in k for p in critical_patterns)]
            old_global_snapshot = {k: global_w[k].detach().clone().float()
                                   for k in actual_layers}
            layer_l2 = {}
            grad_l2 = {}
            for net_id in online_clients:
                client_sd = self.nets_list[net_id].state_dict()
                client_drift = {}
                sq_sum = 0.0
                for layer in actual_layers:
                    if layer in client_sd:
                        diff = client_sd[layer].detach().float() - old_global_snapshot[layer]
                        l2 = float(diff.norm().item())
                        client_drift[layer] = l2
                        sq_sum += l2 ** 2
                layer_l2[net_id] = client_drift
                grad_l2[net_id] = sq_sum ** 0.5
            self._last_layer_l2 = layer_l2
            self._last_grad_l2 = grad_l2
        except Exception:
            self._last_layer_l2 = {}
            self._last_grad_l2 = {}

        # === LAB freq 计算 ===
        lab_used = False
        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[k] for k in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            online_clients_all = float(np.sum(online_clients_len))

            if (self._next_round_lab_freq is not None
                    and not self._next_round_lab_result.get("fallback_to_fedavg", False)):
                w_proj_dom = self._next_round_lab_result["w_proj"]
                cli_freq = self._domain_level_to_cli_freq(w_proj_dom, online_clients)
                if cli_freq is not None and len(cli_freq) == len(online_clients):
                    freq = cli_freq
                    lab_used = True
            if not lab_used:
                freq = np.array(online_clients_len) / online_clients_all
        else:
            parti_num = len(online_clients)
            freq = np.array([1.0 / parti_num for _ in range(parti_num)], dtype=np.float64)

        self._last_lab_used = bool(lab_used)
        self._last_lab_cli_freq = list(map(float, freq))

        # === state_dict 全 key 聚合 ===
        first = True
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)

        # === Sync back to clients ===
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    # ============================================================
    # LAB diag 字段汇总 (merge into proto_logs[-1])
    # ============================================================
    def _compute_lab_round_diag(self, val_loss_per_dom, val_loss_sum_per_cli, val_n_per_cli):
        """组装 LAB 完整诊断 dict (lab_ratio/lab_boost/lab_clip/val_class_counts 等),
        给 utils/diagnostic.py 自动展开 dump 到 npz proto_diag_lab_*."""
        if self._next_round_lab_result is None:
            return {}

        sample_share_dom = self._compute_sample_share_dom(self.online_clients)
        if not sample_share_dom:
            return {}

        positive_delta = self.lab_state.update_boost_record(
            round_idx=self.epoch_index,
            sample_share_dom=sample_share_dom,
            result=self._next_round_lab_result,
        )

        diag = self.lab_state.full_diagnostic(
            round_idx=self.epoch_index,
            sample_share_dom=sample_share_dom,
            lab_result=self._next_round_lab_result,
            positive_delta=positive_delta,
            val_n_per_dom=(self.val_meta or {}).get("val_n_per_dom"),
            val_class_counts=(self.val_meta or {}).get("val_class_counts"),
            val_loss_sum_per_cli=val_loss_sum_per_cli,
            val_n_per_cli=val_n_per_cli,
            signal_round=max(0, self.epoch_index - 1),
        )

        diag["lab_used_this_round"] = 1.0 if getattr(self, "_last_lab_used", False) else 0.0
        if hasattr(self, "_last_lab_cli_freq"):
            for i, k in enumerate(self.online_clients):
                diag[f"lab_cli_freq_{k}"] = float(self._last_lab_cli_freq[i])

        return diag

    # ============================================================
    # Stdout 打印 / waste warning (跟 F2DCPgLab 一致)
    # ============================================================
    def _print_lab_stdout(self, round_idx, result):
        if result.get("fallback_to_fedavg", False):
            print(f"[LAB R{round_idx:3d}] fallback to FedAvg ({result.get('fallback_reason', 'unknown')})")
            return
        domains = result["domains"]
        boost_str = ", ".join(f"{d}:{result['gap'][d]:.3f}" for d in domains)
        ratio_str = ", ".join(f"{d}×{result['ratio'][d]:.2f}" for d in domains)
        loss_str = ", ".join(f"{d}:{result['loss_ema'][d]:.3f}" for d in domains)
        clip_active = [d for d in domains if result['clip_status'][d] is not None]
        clip_str = (",".join(f"{d}{result['clip_status'][d]}" for d in clip_active)
                    if clip_active else "None")
        print(f"[LAB R{round_idx:3d}] loss_ema={{{loss_str}}}")
        print(f"          gap={{{boost_str}}}  ratio={{{ratio_str}}}  clip={{{clip_str}}}")

    def _print_waste_warnings(self, round_idx_override=None):
        if self._next_round_lab_result is None:
            return
        round_for_log = round_idx_override if round_idx_override is not None else self.epoch_index
        domains = self._next_round_lab_result["domains"]
        wasted = self.lab_state.detect_waste(round_for_log, domains)
        for d in wasted:
            roi = self.lab_state.compute_window_roi(d, round_for_log)
            cum_b = float(sum(self.lab_state.boost_history[d]))
            roi_str = f"{roi:.2f}" if roi is not None else "n/a"
            print(f"⚠️ [LAB WASTE WARN R{round_for_log:3d}] dom={d} "
                  f"cum_boost={cum_b*100:.1f}% window_roi={roi_str} "
                  f"(< {self.lab_state.waste_roi_threshold} = wasted)")
