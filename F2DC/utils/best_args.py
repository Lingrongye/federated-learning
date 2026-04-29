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
            'local_lr': 0.01,
            'local_batch_size': 64,
            'lmbd': 0.01,
            'fdse_tau': 0.5,
            'fdse_beta': 0.1,
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
            'local_lr': 0.01,
            'local_batch_size': 64,
            'lmbd': 0.01,
            'fdse_tau': 0.5,
            'fdse_beta': 0.1,
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