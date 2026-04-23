from flgo.experiment.logger import BasicLogger
import numpy as np
import copy
import torch
import os

try:
    from diagnostics.per_class_eval import run_diagnostic as _run_diag
except Exception:  # pragma: no cover
    _run_diag = None


class TuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data) > 0:
            self.set_es_key("val_accuracy")
        else:
            self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data) > 0:
            sval = self.server.test(self.server.model, 'val')
            for met_name in sval.keys():
                self.output['val_' + met_name].append(sval[met_name])
        else:
            cvals = [c.test(self.server.model, 'val') for c in self.clients]
            cdatavols = np.array([len(c.val_data) for c in self.clients])
            cdatavols = cdatavols / cdatavols.sum()
            cval_dict = {}
            if len(cvals) > 0:
                for met_name in cvals[0].keys():
                    if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                    for cid in range(len(cvals)):
                        cval_dict[met_name].append(cvals[cid][met_name])
                    self.output['local_val_' + met_name].append(
                        float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                    self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                    self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class FullLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        train_metrics = self.server.global_test(flag='train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum(
                [client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_' + met_name + '_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum(
                [client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        local_test_metrics = self.server.global_test(flag='test')
        for met_name, met_val in local_test_metrics.items():
            self.output['local_test_' + met_name + '_dist'].append(met_val)
            self.output['local_test_' + met_name].append(1.0 * sum(
                [client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_test_' + met_name].append(np.mean(met_val))
            self.output['std_local_test_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class SimpleLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [len(c.train_data) for c in self.clients]
        total_data_vol = sum(local_data_vols)
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_' + met_name + '_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum(
                [client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class SegTuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data) > 0:
            self.set_es_key("val_Dice")
        else:
            self.set_es_key("local_val_Dice")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data) > 0:
            sval = self.server.test(self.server.model, 'val')
            for met_name in sval.keys():
                self.output['val_' + met_name].append(sval[met_name])
        else:
            cvals = [c.test(self.server.model, 'val') for c in self.clients]
            cdatavols = np.array([len(c.val_data) for c in self.clients])
            cdatavols = cdatavols / cdatavols.sum()
            cval_dict = {}
            if len(cvals) > 0:
                for met_name in cvals[0].keys():
                    if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                    for cid in range(len(cvals)):
                        cval_dict[met_name].append(cvals[cid][met_name])
                    self.output['local_val_' + met_name].append(
                        float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                    self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                    self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class PerTuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        cvals = [c.test(c.model, 'val') if c.model is not None else c.test(self.server.model, 'val') for c in
                 self.clients]
        cdatavols = np.array([len(c.val_data) for c in self.clients])
        cdatavols = cdatavols / cdatavols.sum()
        cval_dict = {}
        if len(cvals) > 0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)): cval_dict[met_name].append(cvals[cid][met_name])
                self.output['local_val_' + met_name + '_dist'].append(cval_dict[met_name])
                self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_val_' + met_name].append(float(np.mean(np.array(cval_dict[met_name]))))
                self.output['std_local_val_' + met_name].append(float(np.std(np.array(cval_dict[met_name]))))
                self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class PerRunLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        # calculate weighted averaging of metrics on training datasets across participants
        cvals = [c.test(c.model, 'val') if c.model is not None else c.test(self.server.model, 'val') for c in
                 self.clients]
        ctests = [c.test(c.model, 'test') if c.model is not None else c.test(self.server.model, 'test') for c in
                  self.clients]
        cdatavols = np.array([len(c.train_data) for c in self.clients])
        cdatavols = cdatavols / cdatavols.sum()
        cval_dict = {}
        ctest_dict = {}
        if len(cvals) > 0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)): cval_dict[met_name].append(cvals[cid][met_name])
                self.output['local_val_' + met_name + '_dist'].append(cval_dict[met_name])
                self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_val_' + met_name].append(float(np.mean(np.array(cval_dict[met_name]))))
                self.output['std_local_val_' + met_name].append(float(np.std(np.array(cval_dict[met_name]))))
                self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        if len(ctests) > 0:
            for met_name in ctests[0].keys():
                if met_name not in ctest_dict.keys(): ctest_dict[met_name] = []
                for cid in range(len(ctests)): ctest_dict[met_name].append(ctests[cid][met_name])
                self.output['local_test_' + met_name + '_dist'].append(ctest_dict[met_name])
                self.output['local_test_' + met_name].append(float((np.array(ctest_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_test_' + met_name].append(float(np.mean(np.array(ctest_dict[met_name]))))
                self.output['std_local_test_' + met_name].append(float(np.std(np.array(ctest_dict[met_name]))))
                self.output['min_local_test_' + met_name].append(float(np.array(ctest_dict[met_name]).min()))
                self.output['max_local_test_' + met_name].append(float(np.array(ctest_dict[met_name]).max()))
        self.show_current_output()
        # train_metrics = self.server.global_test(flag='train')
        # for met_name, met_val in train_metrics.items():
        #     self.output['train_' + met_name + '_dist'].append(met_val)
        #     self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        # local_val_metrics = self.server.global_test(flag='val')
        # for met_name, met_val in local_val_metrics.items():
        #     self.output['local_val_'+met_name+'_dist'].append(met_val)
        #     self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        #     self.output['mean_local_val_' + met_name].append(np.mean(met_val))
        #     self.output['std_local_val_' + met_name].append(np.std(met_val))
        # local_test_metrics = self.server.global_test(flag='test')
        # for met_name, met_val in local_test_metrics.items():
        #     self.output['local_test_'+met_name+'_dist'].append(met_val)
        #     self.output['local_test_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        #     self.output['mean_local_test_' + met_name].append(np.mean(met_val))
        #     self.output['std_local_test_' + met_name].append(np.std(met_val))
        # output to stdout

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class PerRunDiagLogger(PerRunLogger):
    """PerRunLogger + per-class + confidence diagnostic on every round.

    Records same fields as PerRunLogger, plus:
      per_class_test_acc_dist[round][client][class] = float
      per_class_test_conf_dist[round][client][class] = float
      per_class_test_support_dist[round][client][class] = int
      confidence_stats_dist[round][client] = {mean, std, p10, p50, p90,
                                              ece, over_conf_err_ratio,
                                              wrong_conf_mean}
      (every HIST_EVERY rounds) confidence_hist_dist[round][client] =
                                              {hist_correct, hist_wrong, bins}

    Runs eval once per client per round. Overhead: +1 forward pass on test
    set per round (model already runs test once for PerRunLogger metrics,
    so we do a separate forward here to get softmax + labels).

    Number of classes is inferred from the first client's model output on
    the first test batch.
    """

    HIST_EVERY = 50  # dump full histogram every this-many rounds

    def _get_num_classes(self) -> int:
        """Infer num_classes from the task's known class set."""
        # PACS=7, OfficeCaltech10=10, DomainNet=?, Digit5=10
        # Try benchmark name heuristic; fall back to model output.
        task_name = getattr(self.server, 'task', '') or ''
        task_name = task_name.lower()
        if 'pacs' in task_name:
            return 7
        if 'office' in task_name:
            return 10
        if 'digit' in task_name:
            return 10
        if 'domainnet' in task_name:
            return 10  # DomainNet-10 variant used here
        # Fallback: find a Linear at the end of the model with N output features
        model = self.server.model
        last_linear_out = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear_out = m.out_features
        if last_linear_out is None:
            raise RuntimeError("Cannot infer num_classes for diagnostic logger")
        return int(last_linear_out)

    def log_once(self, *args, **kwargs):
        # First: run parent logic (records local_test_accuracy etc.)
        super().log_once(*args, **kwargs)

        if _run_diag is None:
            return  # diagnostic module not importable — silent skip

        num_classes = self._get_num_classes()
        round_id = len(self.output.get('local_test_accuracy', [])) - 1
        include_hist = (round_id % self.HIST_EVERY == 0) or (round_id == 0)

        # Collect per-client diagnostic dict
        pc_acc_dist: list = []
        pc_conf_dist: list = []
        pc_support_dist: list = []
        conf_stats_dist: list = []
        hist_dist: list = []

        for c in self.clients:
            model = c.model if getattr(c, 'model', None) is not None else self.server.model
            test_data = getattr(c, 'test_data', None)
            if test_data is None or len(test_data) == 0:
                nan_list = [float('nan')] * num_classes
                pc_acc_dist.append(nan_list)
                pc_conf_dist.append(nan_list)
                pc_support_dist.append([0] * num_classes)
                conf_stats_dist.append({})
                if include_hist:
                    hist_dist.append({})
                continue

            # Resolve device from model's current param placement (avoid
            # forcing a device move; parent c.test() may have already set it)
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = self.server.device
            bs = min(getattr(c, 'test_batch_size', 64), len(test_data))
            num_workers = self.option.get('num_workers', 0)
            # Collate fn: taken from client's calculator if present
            collate_fn = getattr(getattr(c, 'calculator', None), 'collate_fn', None)

            try:
                diag = _run_diag(
                    model=model,
                    dataset=test_data,
                    device=device,
                    num_classes=num_classes,
                    batch_size=bs,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    include_histogram=include_hist,
                )
            except Exception as e:
                # Don't crash the run if diagnostic fails — just log zeros
                import traceback
                print(f"[PerRunDiagLogger] client diag failed: {e}")
                traceback.print_exc()
                nan_list = [float('nan')] * num_classes
                diag = {
                    'per_class_acc': nan_list,
                    'per_class_conf': nan_list,
                    'per_class_support': [0] * num_classes,
                    'conf_mean': float('nan'),
                    'conf_std': float('nan'),
                    'conf_p10': float('nan'),
                    'conf_p50': float('nan'),
                    'conf_p90': float('nan'),
                    'ece': float('nan'),
                    'over_conf_err_ratio': float('nan'),
                    'wrong_conf_mean': float('nan'),
                    'overall_acc': float('nan'),
                }

            pc_acc_dist.append(diag['per_class_acc'])
            pc_conf_dist.append(diag['per_class_conf'])
            pc_support_dist.append(diag['per_class_support'])
            conf_stats_dist.append({
                'mean': diag['conf_mean'], 'std': diag['conf_std'],
                'p10': diag['conf_p10'], 'p50': diag['conf_p50'], 'p90': diag['conf_p90'],
                'ece': diag['ece'],
                'over_conf_err_ratio': diag['over_conf_err_ratio'],
                'wrong_conf_mean': diag['wrong_conf_mean'],
            })
            if include_hist:
                hist_dist.append({
                    'hist_correct': diag.get('hist_correct'),
                    'hist_wrong': diag.get('hist_wrong'),
                    'bins': diag.get('hist_bins', 20),
                })

        self.output['per_class_test_acc_dist'].append(pc_acc_dist)
        self.output['per_class_test_conf_dist'].append(pc_conf_dist)
        self.output['per_class_test_support_dist'].append(pc_support_dist)
        self.output['confidence_stats_dist'].append(conf_stats_dist)
        if include_hist:
            self.output['confidence_hist_dist'].append({
                'round': round_id,
                'per_client': hist_dist,
            })


class PerSegTuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_Dice")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        cvals = [c.test(c.model, 'val') if c.model is not None else c.test(self.server.model, 'val') for c in
                 self.clients]
        cdatavols = np.array([len(c.val_data) for c in self.clients])
        cdatavols = cdatavols / cdatavols.sum()
        cval_dict = {}
        if len(cvals) > 0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)): cval_dict[met_name].append(cvals[cid][met_name])
                self.output['local_val_' + met_name + '_dist'].append(cval_dict[met_name])
                self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_val_' + met_name].append(float(np.mean(np.array(cval_dict[met_name]))))
                self.output['std_local_val_' + met_name].append(float(np.std(np.array(cval_dict[met_name]))))
                self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class PerSegRunLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_Dice")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        # calculate weighted averaging of metrics on training datasets across participants
        cvals = [c.test(c.model, 'val') if c.model is not None else c.test(self.server.model, 'val') for c in
                 self.clients]
        ctests = [c.test(c.model, 'test') if c.model is not None else c.test(self.server.model, 'test') for c in
                  self.clients]
        cdatavols = np.array([len(c.train_data) for c in self.clients])
        cdatavols = cdatavols / cdatavols.sum()
        cval_dict = {}
        ctest_dict = {}
        if len(cvals) > 0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)): cval_dict[met_name].append(cvals[cid][met_name])
                self.output['local_val_' + met_name + '_dist'].append(cval_dict[met_name])
                self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_val_' + met_name].append(float(np.mean(np.array(cval_dict[met_name]))))
                self.output['std_local_val_' + met_name].append(float(np.std(np.array(cval_dict[met_name]))))
                self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        if len(ctests) > 0:
            for met_name in ctests[0].keys():
                if met_name not in ctest_dict.keys(): ctest_dict[met_name] = []
                for cid in range(len(ctests)): ctest_dict[met_name].append(ctests[cid][met_name])
                self.output['local_test_' + met_name + '_dist'].append(ctest_dict[met_name])
                self.output['local_test_' + met_name].append(float((np.array(ctest_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_test_' + met_name].append(float(np.mean(np.array(ctest_dict[met_name]))))
                self.output['std_local_test_' + met_name].append(float(np.std(np.array(ctest_dict[met_name]))))
                self.output['min_local_test_' + met_name].append(float(np.array(ctest_dict[met_name]).min()))
                self.output['max_local_test_' + met_name].append(float(np.array(ctest_dict[met_name]).max()))
        self.show_current_output()

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res


class DALogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)

    def log_once(self, *args, **kwargs):
        # calculate weighted averaging of metrics on training datasets across participants
        cvals = [c.test(c.model, 'val') if c.model is not None else c.test(self.server.model, 'val') for c in
                 self.clients]
        ctests = [c.test(c.model, 'test') if c.model is not None else c.test(self.server.model, 'test') for c in
                  self.clients]
        cdatavols = np.array([len(c.train_data) for c in self.clients])
        cdatavols = cdatavols / cdatavols.sum()
        cval_dict = {}
        ctest_dict = {}
        if len(cvals) > 0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)): cval_dict[met_name].append(cvals[cid][met_name])
                self.output['local_val_' + met_name + '_dist'].append(cval_dict[met_name])
                self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_val_' + met_name].append(float(np.mean(np.array(cval_dict[met_name]))))
                self.output['std_local_val_' + met_name].append(float(np.std(np.array(cval_dict[met_name]))))
                self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        if len(ctests) > 0:
            for met_name in ctests[0].keys():
                if met_name not in ctest_dict.keys(): ctest_dict[met_name] = []
                for cid in range(len(ctests)): ctest_dict[met_name].append(ctests[cid][met_name])
                self.output['local_test_' + met_name + '_dist'].append(ctest_dict[met_name])
                self.output['local_test_' + met_name].append(float((np.array(ctest_dict[met_name]) * cdatavols).sum()))
                self.output['mean_local_test_' + met_name].append(float(np.mean(np.array(ctest_dict[met_name]))))
                self.output['std_local_test_' + met_name].append(float(np.std(np.array(ctest_dict[met_name]))))
                self.output['min_local_test_' + met_name].append(float(np.array(ctest_dict[met_name]).min()))
                self.output['max_local_test_' + met_name].append(float(np.array(ctest_dict[met_name]).max()))
        self.show_current_output()
        # train_metrics = self.server.global_test(flag='train')
        # for met_name, met_val in train_metrics.items():
        #     self.output['train_' + met_name + '_dist'].append(met_val)
        #     self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        # local_val_metrics = self.server.global_test(flag='val')
        # for met_name, met_val in local_val_metrics.items():
        #     self.output['local_val_'+met_name+'_dist'].append(met_val)
        #     self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        #     self.output['mean_local_val_' + met_name].append(np.mean(met_val))
        #     self.output['std_local_val_' + met_name].append(np.std(met_val))
        # local_test_metrics = self.server.global_test(flag='test')
        # for met_name, met_val in local_test_metrics.items():
        #     self.output['local_test_'+met_name+'_dist'].append(met_val)
        #     self.output['local_test_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        #     self.output['mean_local_test_' + met_name].append(np.mean(met_val))
        #     self.output['std_local_test_' + met_name].append(np.std(met_val))
        # output to stdout

    def optimal_state(self) -> dict:
        res = {'model': copy.deepcopy(self.coordinator.model.state_dict())}
        if hasattr(self.clients[0], 'model') and self.clients[0].model is not None:
            res.update({'local_models': [copy.deepcopy(c.model.state_dict()) if c.model is not None else {} for c in
                                         self.clients]})
        return res

    def save_optimal_state(self):
        if self._optimal_state is not None: torch.save(self._optimal_state, os.path.join(self.get_output_path(),
                                                                                         f"DA-S{self.option['seed']}-" + self.get_output_name(
                                                                                             suffix='.pth')))