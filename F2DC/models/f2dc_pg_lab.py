"""
F2DC-PG + LAB v4.2 (Loss-Aware Boost) Aggregation
======================================================

继承 F2DCPG (= F2DC + Prototype-Guided DFC). 唯一改动: 把 server 端
sample-weighted 聚合 (FedAvg) / DaA Eq(10/11) 替换为 LAB v4.2:

  client 持本地 held-out val (从 partition unused 反推) → 上传 (loss_sum, val_n)
  → server sample-weighted 聚合 val_loss → ReLU(loss - mean) underfit boost
  → λ-mix + bounded simplex projection → 域内等权 → 聚合权重 freq.

设计原则:
1. 不改原有文件 (federated_model.py / partition / utils/training.py 都不动)
2. 完整继承 F2DCPG, 只覆盖 aggregate_nets + 在 loc_update 末尾加 val_loss eval
3. LAB 诊断 (20+ 字段) 通过 model.proto_logs[-1] 自动被 utils/diagnostic.py dump

接入:
    --model f2dc_pg_lab \
    --use_daa false \                     # 跟 LAB 互斥
    --lab_lambda 0.15 \                   # PROPOSAL 预注册
    --lab_ratio_min 0.80 \
    --lab_ratio_max 2.00 \
    --lab_ema_alpha 0.30 \
    --lab_val_size_per_dom 50 \
    --lab_val_per_class 5 \
    --lab_val_seed 42

PROPOSAL: experiments/ablation/EXP-144_lab_v4_1/PROPOSAL.md
"""
import copy
import numpy as np
import torch

from models.f2dc_pg import F2DCPG
from models.utils.lab_aggregation import LabState
from datasets.utils.lab_partition import setup_lab_val_loaders, evaluate_val_per_client


class F2DCPgLab(F2DCPG):
    """F2DC + PG-DFC + LAB aggregation (替代 DaA / FedAvg sample-weighted)."""
    NAME = 'f2dc_pg_lab'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)

        # ===== LAB 超参 (PROPOSAL §10) =====
        self.lab_lambda = float(getattr(args, 'lab_lambda', 0.15))
        self.lab_ratio_min = float(getattr(args, 'lab_ratio_min', 0.80))
        self.lab_ratio_max = float(getattr(args, 'lab_ratio_max', 2.00))
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
        )

        # val_loaders 在 first round 才 setup (因为 trainloaders / selected_domain_list
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
            print("[LAB WARNING] --use_daa=True is incompatible with f2dc_pg_lab; "
                  "DaA is auto-disabled, using LAB instead.")
            self.args.use_daa = False

    # ============================================================
    # Per-round test acc hook (utils/training.py 在 evaluate 之后调用)
    # ============================================================
    def lab_record_test_acc(self, round_idx: int, accs, all_dataset_names) -> None:
        """utils/training.py global_evaluate 完后调一次, 把 per-domain test acc 喂给 LabState.

        必要性 (修 codex Important #5): LabState.acc_history 之前永远空,
        ROI/waste 完全不可用. 现在让 utils/training.py 显式喂.

        Patch (codex 三轮 Important #1): 因为 loc_update() 内的 full_diagnostic()
        在本函数之前调用, 它写入 proto_logs[-1] 的 lab_delta_acc_* 用的是 acc_history[-1] - acc_history[-2],
        此时 acc_history 还没本 round 的 acc, 所以记录的是 (acc[R-1] - acc[R-2]) 即滞后 1 round.
        现在更新完 acc_history 后, 立刻 patch proto_logs[-1] 的 lab_delta_acc_*
        让它反映本 round 真实 (acc[R] - acc[R-1]).

        参数:
            round_idx: 1-based round (= epoch_index + 1)
            accs: list of per-domain test acc (跟 all_dataset_names 一一对应)
            all_dataset_names: ['photo', 'art', 'cartoon', 'sketch'] 等
        """
        if not hasattr(self, "lab_state") or self.lab_state is None:
            return
        try:
            acc_dict = {str(d): float(a) for d, a in zip(all_dataset_names, accs)}
            self.lab_state.update_test_acc(round_idx, acc_dict)

            # Patch (codex 三轮 Important #1): 重写本 round proto_logs[-1].lab_delta_acc_*
            if hasattr(self, "proto_logs") and self.proto_logs:
                latest = self.proto_logs[-1]
                if isinstance(latest, dict):
                    for d in acc_dict:
                        ah = self.lab_state.acc_history.get(d, [])
                        if len(ah) >= 2:
                            latest[f"lab_delta_acc_{d}"] = float(ah[-1] - ah[-2])
                        else:
                            latest[f"lab_delta_acc_{d}"] = 0.0   # round 1: 没有上一轮

            # Live waste warning (codex 四轮 Important fix): 移到这里, evaluate 后才有真实 acc
            # 用 round_idx (= epoch_index + 1) 当 1-based round 计数
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
        """在第一次 loc_update 调用时跑 (此时 trainloaders / _selected_domain_list 已 ready)."""
        if self._lab_setup_done:
            return
        if not hasattr(self, 'trainloaders') or self.trainloaders is None:
            print("[LAB ERROR] trainloaders not yet set, val pool setup deferred.")
            return
        if not hasattr(self, '_selected_domain_list') or self._selected_domain_list is None:
            print("[LAB ERROR] _selected_domain_list not set; val pool setup deferred.")
            return

        # 拿每个 cli 的 train_dataset (从 trainloader.dataset)
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
            self.val_meta = {"val_n_per_cli": {}, "val_n_per_dom": {},
                             "val_class_counts": {}}
            self._lab_setup_done = True
            return

        self.val_loaders = val_loaders
        self.val_meta = val_meta
        self._lab_setup_done = True

        # Stdout 打印 setup summary
        n_total = sum(val_meta["val_n_per_cli"].values())
        n_active_cli = len(val_meta["val_n_per_cli"])
        print(f"[LAB setup] val partition: {n_total} samples across {n_active_cli} clients, "
              f"per-dom = {val_meta['val_n_per_dom']}")
        for d, cls in val_meta.get("val_class_counts", {}).items():
            print(f"           {d} class counts: {dict(sorted(cls.items()))}")

    # ============================================================
    # 主 round 循环 (override F2DCPG.loc_update)
    # ============================================================
    def loc_update(self, priloader_list):
        # First-round val setup
        self._ensure_lab_setup()

        # === 计算本 round LAB freq (基于上一轮 val_loss EMA) ===
        # 必须在 client 训练 + aggregate 之前算, 因为 aggregate 时直接读
        # self._next_round_lab_freq + self._next_round_lab_result.
        # (round 1 时 _next_round_lab_freq=None, fallback FedAvg sample-weighted.)

        # 接下来跟 F2DCPG.loc_update 完全一致 (不动 prototype 逻辑) ↓↓↓
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []

        # 当前 round 的 proto_weight (跟 F2DCPG 一致)
        current_pw = self._get_proto_weight()

        # 把 proto_weight 同步到所有 client model + 下发 global proto buffer
        for i in online_clients:
            net = self.nets_list[i]
            if hasattr(net, 'dfc_module') and hasattr(net.dfc_module, 'set_proto_weight'):
                net.dfc_module.set_proto_weight(current_pw)
                if self.global_proto_unit is not None:
                    net.dfc_module.class_proto.copy_(self.global_proto_unit.to(self.device))
                if hasattr(net.dfc_module, 'reset_diag'):
                    net.dfc_module.reset_diag()
                if hasattr(net, 'dfd_module') and hasattr(net.dfd_module, 'reset_diag'):
                    net.dfd_module.reset_diag()

        # 训练所有 online clients
        round_diag_collect = []
        for i in online_clients:
            c_loss, c_samples = self._train_net_pg(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss
            self.num_samples.append(c_samples)
            net = self.nets_list[i]
            d = {}
            if hasattr(net, 'dfd_module') and hasattr(net.dfd_module, 'get_diag_summary'):
                dfd_d = net.dfd_module.get_diag_summary()
                if dfd_d is not None:
                    d.update(dfd_d)
            if hasattr(net, 'dfc_module') and hasattr(net.dfc_module, 'get_diag_summary'):
                dfc_d = net.dfc_module.get_diag_summary()
                if dfc_d is not None:
                    for k, v in dfc_d.items():
                        if k.startswith('attn_') or k.startswith('proto_'):
                            d[k] = v
                    if 'mask_mean_mean' not in d:
                        for k, v in dfc_d.items():
                            if k.startswith('mask_'):
                                d[k] = v
            if d:
                d['client_id'] = i
                round_diag_collect.append(d)

        # === LAB-replaced backbone 聚合 ===
        # aggregate_nets 内部会优先用 self._next_round_lab_freq;
        # round 1 (None) 时 fallback FedAvg sample-weighted (跟 vanilla 一致).
        self.aggregate_nets(None)

        # === Prototype 聚合 (PG-DFC v3.2, 不动) ===
        if self.epoch_index >= self.warmup_rounds - 1:
            self.aggregate_protos_v3()

        self._sync_global_proto_to_global_net(current_pw)

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
                # 写入 LabState (给下一轮 LAB 用)
                self.lab_state.update_val_loss(self.epoch_index, val_loss_per_dom)
            except Exception as e:
                print(f"[LAB val eval ERR] round {self.epoch_index}: {e}")

        # === LAB diag 写入 proto_logs (utils/diagnostic.py 自动 dump) ===
        round_summary = self._summarize_round_diag(round_diag_collect)
        round_summary['round'] = self.epoch_index
        round_summary['proto_weight_active'] = current_pw
        if self.global_proto_unit is not None:
            round_summary['global_proto_norm_mean'] = float(
                self.global_proto_unit.norm(dim=-1).mean().item()
            )

        # 把本 round LAB freq 计算结果 (在 aggregate_nets 内部已存) 加进 diag
        lab_diag = self._compute_lab_round_diag(
            val_loss_per_dom=val_loss_per_dom,
            val_loss_sum_per_cli=val_loss_sum_per_cli,
            val_n_per_cli=val_n_per_cli,
        )
        if lab_diag:
            round_summary.update(lab_diag)

        self.proto_logs.append(round_summary)

        # === stdout 实时打印 (PROPOSAL §5.3) ===
        if self.lab_print_diag and self._next_round_lab_result is not None:
            self._print_lab_stdout(self.epoch_index, self._next_round_lab_result)

        # 注意 (codex 四轮 Important): waste warning 移到 lab_record_test_acc() 末尾,
        # 因为它依赖 acc_history (本 round 的 acc 在 evaluate 之后才更新).
        # 在 loc_update 内调会用滞后 1 round 的数据.

        # === 为下一 round 预算 LAB freq ===
        # (本 round 已经聚合完, 下一 round loc_update 进 aggregate_nets 前会用)
        self._precompute_next_lab_freq()

        all_c_avg_loss = all_clients_loss / max(len(online_clients), 1)
        return round(all_c_avg_loss, 3)

    # ============================================================
    # LAB freq 预算 (给下一 round 用)
    # ============================================================
    def _precompute_next_lab_freq(self):
        """本 round 末调用. 用 LabState.val_loss_ema (刚刚更新) + sample_share_dom
        算下一轮要用的 freq. 存到 self._next_round_lab_freq / _next_round_lab_result."""
        if not hasattr(self, 'trainloaders') or self.trainloaders is None:
            self._next_round_lab_freq = None
            self._next_round_lab_result = None
            return

        # 算 sample_share_dom (基于本 round online_clients)
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

        # 把 domain-level w_proj 转成 cli-level freq (基于下一轮 online_clients;
        # 但下一轮 online_clients 还没决定, 这里用 fixed allocation 假设 full participation)
        # 因为 fixed allocation 下 selected_domain_list 是固定的, 所以 cli-level 映射是
        # confidence 的: 每个 cli 持单一 domain, 域内等权.
        self._next_round_lab_freq = result   # 把 domain-level w_proj 存着, aggregate_nets 时再 cli 级展开

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
        # 每域 cli 数 (在 online_clients 里)
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

        # 防御 normalize (理论上 w_proj 已 sum to 1, 但 online_clients 可能少)
        s = freq.sum()
        if s > 1e-9:
            freq = freq / s
        return freq

    # ============================================================
    # Override aggregate_nets — 用 LAB freq 替换 DaA / FedAvg
    # ============================================================
    def aggregate_nets(self, freq=None):
        """
        Override federated_model.FederatedModel.aggregate_nets.
        Logic:
          - 复制基类的 grad_l2 / layer_l2 诊断 hook (一字不动)
          - freq 用 LAB (如有) 否则 fallback FedAvg sample-weighted
          - state_dict 全 key 聚合 + 同步回 client (跟 F2DC release 一致)
        """
        global_net = self.global_net
        nets_list = self.nets_list
        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        # === diagnostic: 在 sync 前算 client 跟 global 的 drift (跟基类一致) ===
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
        # 提前初始化 lab_used (避免作用域跨 if/else 出问题)
        lab_used = False
        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[k] for k in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            online_clients_all = float(np.sum(online_clients_len))

            if (self._next_round_lab_freq is not None
                    and not self._next_round_lab_result.get("fallback_to_fedavg", False)):
                # LAB 路径
                w_proj_dom = self._next_round_lab_result["w_proj"]
                cli_freq = self._domain_level_to_cli_freq(w_proj_dom, online_clients)
                if cli_freq is not None and len(cli_freq) == len(online_clients):
                    freq = cli_freq
                    lab_used = True
            if not lab_used:
                # Fallback FedAvg (round 1 / setup error / val 全空 / 缺 domain)
                freq = np.array(online_clients_len) / online_clients_all
        else:
            parti_num = len(online_clients)
            freq = np.array([1.0 / parti_num for _ in range(parti_num)], dtype=np.float64)

        # 暴露给 diagnostic hook (用真实 lab_used 局部变量)
        self._last_lab_used = bool(lab_used)
        self._last_lab_cli_freq = list(map(float, freq))

        # === state_dict 全 key 聚合 (跟基类一致) ===
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

        # === Sync back to clients (跟基类一致) ===
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    # ============================================================
    # LAB diag 字段汇总 (写入 proto_logs)
    # ============================================================
    def _compute_lab_round_diag(self, val_loss_per_dom, val_loss_sum_per_cli, val_n_per_cli):
        """组装 LAB 完整诊断 dict, 含 20+ 字段, 给 utils/diagnostic.py 自动展开 dump."""
        if self._next_round_lab_result is None:
            return {}

        # 算 sample_share_dom (用本 round online_clients)
        sample_share_dom = self._compute_sample_share_dom(self.online_clients)
        if not sample_share_dom:
            return {}

        # 计算 positive_delta + 写入 boost_history
        positive_delta = self.lab_state.update_boost_record(
            round_idx=self.epoch_index,
            sample_share_dom=sample_share_dom,
            result=self._next_round_lab_result,
        )

        # 把当前 test acc 也喂给 LabState (utils/training.py 还没把 acc 传过来,
        # 这步在 dump_round_metadata 之前; 我们暂时用空 dict, 等 dump_round_metadata
        # 把 per_domain_acc 经 model.proto_logs 反向拿到再补).
        # PG-DFC 流程里 acc 在 evaluate 之后才有, 所以这里 acc_history 滞后一轮.
        # 不影响 LAB 本身, 只影响 ROI 计算 — ROI 反正也是后期诊断.

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

        # 加 lab_used / lab_cli_freq 标识
        diag["lab_used_this_round"] = 1.0 if getattr(self, "_last_lab_used", False) else 0.0
        if hasattr(self, "_last_lab_cli_freq"):
            for i, k in enumerate(self.online_clients):
                diag[f"lab_cli_freq_{k}"] = float(self._last_lab_cli_freq[i])

        return diag

    # ============================================================
    # Stdout 打印 / waste warning
    # ============================================================
    def _print_lab_stdout(self, round_idx, result):
        if result.get("fallback_to_fedavg", False):
            print(f"[LAB R{round_idx:3d}] fallback to FedAvg ({result.get('fallback_reason', 'unknown')})")
            return
        domains = result["domains"]
        boost_str = ", ".join(
            f"{d}:{result['gap'][d]:.3f}" for d in domains
        )
        ratio_str = ", ".join(
            f"{d}×{result['ratio'][d]:.2f}" for d in domains
        )
        loss_str = ", ".join(
            f"{d}:{result['loss_ema'][d]:.3f}" for d in domains
        )
        clip_active = [d for d in domains if result['clip_status'][d] is not None]
        clip_str = (",".join(f"{d}{result['clip_status'][d]}" for d in clip_active)
                    if clip_active else "None")
        print(f"[LAB R{round_idx:3d}] loss_ema={{{loss_str}}}")
        print(f"          gap={{{boost_str}}}  ratio={{{ratio_str}}}  clip={{{clip_str}}}")

    def _print_waste_warnings(self, round_idx_override=None):
        """打印 waste warning. round_idx_override 用于 lab_record_test_acc 时传 evaluate 后的 round.

        如果不传 override, 用 self.epoch_index (loc_update 内用, 但此时 acc_history 滞后).
        codex 四轮 Important: 从 loc_update 移到 lab_record_test_acc 末尾, 用 evaluate 后的真实 acc.
        """
        if self._next_round_lab_result is None:
            return
        # 优先用 lab_state.last_round_seen (lab_record_test_acc 更新过), 否则 fallback
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
