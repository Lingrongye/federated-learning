VERDICT: REVISE

**CRITICAL**
- [feddsa_sgpa.py:64](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L64) and [feddsa_sgpa.py:689](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L689): the warmup is not actually “baseline-equivalent”. `lambda_adv=0` only zeroes the encoder gradient on the `z_sem` path via GRL; it does not disable `L_dom_sem`, and `L_dom_sty` still fully updates both `style_head` and `dom_head` from round 0. So rounds 0..20 are already running CDANN pressure, contrary to the schedule comment and your stated intent. If warmup is meant to freeze the CDANN branch, gate the domain losses themselves during warmup, not just the GRL coefficient.

**IMPORTANT**
- [feddsa_sgpa.py:1012](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L1012) and [feddsa_sgpa.py:315](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L315): `dom_head` output size is inferred from `_TASK_NUM_CLIENTS` task-name heuristics, while the actual runtime count is separately computed as `len(self.clients)` and never used to size the model. Any custom split/task alias/mismatch can produce `client.id >= num_clients`, which will break `cross_entropy` or silently encode the wrong domain space.
- [feddsa_sgpa.py:267](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L267), [feddsa_sgpa.py:689](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L689), and [feddsa_sgpa.py:290](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L290): code comments say `ca=1` “requires `use_etf=0`”, but there is no enforcement. If someone enables `ca=1, ue=1`, CDANN still runs, and diagnostics are routed to `etf` rather than `cdann`. That is a real config footgun.

**MINOR**
- [feddsa_sgpa.py:316](D:/桌面文件/联邦学习/FDSE_CVPR25/algorithm/feddsa_sgpa.py#L316): `c.num_clients_total` is dead state in this diff. Either use it to size/validate the domain head or drop it.

No actionable issue found with “constant per-client domain label per batch” itself, and `dom_head` is included in FedAvg by the current shared-key filter.