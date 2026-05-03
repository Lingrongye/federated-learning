best_args = {
    'fl_digits': {
        'fedavg': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedopt': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'global_lr': 0.25,
        },
        'fedproc': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu': 0.01,
        },
        'fedproto': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu':1.0
        },
        'feddyn': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'reg_lamb': 0.5
        },
        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'temperature': 0.5,
                'mu':5
        },
        'f2dc': {
            'local_lr': 0.01,
            'local_batch_size': 64
        },
        'fdse': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'lmbd': 0.01,
            'fdse_tau': 0.5,
            'fdse_beta': 0.1,
        }
    },

    'fl_officecaltech': {
        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'fedproc': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedopt': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'global_lr': 0.25,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 64,
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'temperature': 0.5,
            'mu': 5
        },
        'fedproto': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu':1.0
        },
        'feddyn': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'reg_lamb': 0.5
        },
        'f2dc': {
            'local_lr': 0.01,
            'local_batch_size': 64
        },
        'fdse': {
            # Office 专用 hyperparams: 跟 FDSE_CVPR25/task/office_caltech10_c4/log/*algopara_0.050.50.05*
            # 文件名解码 (lmbd=0.05, tau=0.5, beta=0.05) 一致, 替代默认 lmbd=0.01/beta=0.1
            'local_lr': 0.01,
            'local_batch_size': 64,
            'lmbd': 0.05,
            'fdse_tau': 0.5,
            'fdse_beta': 0.05,
        }
    },

    'fl_pacs': {
        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 46,
        },
        'fedopt': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'global_lr': 0.25,
        },
        'fedproc': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 64,
        },
        'fedproc': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'temperature': 0.5,
            'mu': 5
        },
        'fedproto': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu':1.0
        },
        'feddyn': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'reg_lamb': 0.5
        },
        'f2dc': {
            'local_lr': 0.01,
            'local_batch_size': 64
        },
        'f2dc_pg': {
            'local_lr': 0.01,
            'local_batch_size': 64
        },
        'fdse': {
            # PACS 专用 hyperparams: 跟 FDSE_CVPR25/task/PACS_c4/log/*algopara_0.50.50.001*
            # 文件名解码 (lmbd=0.5, tau=0.5, beta=0.001) 一致, 修复 lmbd=0.01/beta=0.1 默认值
            # 在 PACS 上训崩 (R18 peak 53.40 → R99 33.58 collapse) 的 bug.
            'local_lr': 0.01,
            'local_batch_size': 64,
            'lmbd': 0.5,
            'fdse_tau': 0.5,
            'fdse_beta': 0.001,
        }
    }
}

# Inject f2dc_pg into all datasets (mirror f2dc default per-dataset batch_size)
for _ds in best_args:
    if 'f2dc_pg' not in best_args[_ds]:
        best_args[_ds]['f2dc_pg'] = dict(best_args[_ds].get('f2dc', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject f2dc_pgv33 (mirror f2dc_pg defaults)
for _ds in best_args:
    if 'f2dc_pgv33' not in best_args[_ds]:
        best_args[_ds]['f2dc_pgv33'] = dict(best_args[_ds].get('f2dc_pg', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject f2dc_pg_ml (mirror f2dc_pgv33 defaults)
for _ds in best_args:
    if 'f2dc_pg_ml' not in best_args[_ds]:
        best_args[_ds]['f2dc_pg_ml'] = dict(best_args[_ds].get('f2dc_pgv33', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject f2dc_pg_lab (EXP-144 LAB v4.2 — mirror f2dc_pg defaults, LAB 替代 DaA 聚合)
for _ds in best_args:
    if 'f2dc_pg_lab' not in best_args[_ds]:
        best_args[_ds]['f2dc_pg_lab'] = dict(best_args[_ds].get('f2dc_pg', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject f2dc_dse_lab (EXP-149 — mirror f2dc_dse defaults, LAB 替代 DaA 聚合 + DSE adapter)
for _ds in best_args:
    if 'f2dc_dse_lab' not in best_args[_ds]:
        best_args[_ds]['f2dc_dse_lab'] = dict(best_args[_ds].get('f2dc_dse', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject f2dc_dse (mirror f2dc defaults: 纯 F2DC base + layer3 DSE_Rescue3)
for _ds in best_args:
    if 'f2dc_dse' not in best_args[_ds]:
        best_args[_ds]['f2dc_dse'] = dict(best_args[_ds].get('f2dc', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject fedbn into all datasets (mirror fedavg, no extra hyperparams)
for _ds in best_args:
    if 'fedbn' not in best_args[_ds]:
        best_args[_ds]['fedbn'] = dict(best_args[_ds].get('fedavg', {'local_lr': 0.01, 'local_batch_size': 64}))

# Inject fedprox / fedproto for all datasets if missing (some only registered for fl_pacs)
for _ds in best_args:
    if 'fedprox' not in best_args[_ds]:
        _base = dict(best_args[_ds].get('fedavg', {'local_lr': 0.01, 'local_batch_size': 64}))
        _base['mu'] = 0.01
        best_args[_ds]['fedprox'] = _base
    if 'fedproto' not in best_args[_ds]:
        _base = dict(best_args[_ds].get('fedavg', {'local_lr': 0.01, 'local_batch_size': 64}))
        _base['mu'] = 1.0
        best_args[_ds]['fedproto'] = _base