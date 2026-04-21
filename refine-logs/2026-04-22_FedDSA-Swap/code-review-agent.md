# EXP-113 Full Review (Agent)

Scope: `algorithm/common/{vib,supcon,diagnostic_ext}.py`, `algorithm/feddsa_sgpa_vib.py`, 6 configs. Parent kept intact.

## Category A: Bugs (Crash / NaN)

### A1. Numerical stability over 200 rounds — **Minor**
- `log_var` clamp `[-5, 2]` → σ ∈ `[0.082, 2.72]`. OK.
- KL closed-form uses `var_p = exp(log_var_p)` where `log_var_p = 2 * log_sigma_prior[y]`. `log_sigma_prior` is unclamped (learnable). If it drifts to e.g. `-10`, `var_p → 2e-9`, making `(mu_q - mu_p)² / var_p` explode. Smoke R5 kl_mean=23.70 (fine), but over 200 rounds with CE pressure this can blow up. **Fix suggestion**: add `log_sigma_prior` clamp in `VIBSemanticHead.forward` (e.g. `torch.clamp(self.log_sigma_prior[y], -3, 2)`).
- Server EMA β=0.99 is conservative; prototype won't oscillate.

### A2. Prototype EMA convergence — **Minor**
EMA applied only to classes present in ≥1 client this round. With full sampling (`proportion: 1.0`), all 4 clients participate each round, so all 7 PACS / 10 Office classes get updates. Convergence is fine. No issue.

### A3. `private_keys` coverage — **None (OK)**
`_init_agg_keys` correctly adds `style_head`, BN running stats, `log_var_head`, `log_sigma_prior`, `prototype_ema`, `prototype_init`, and `M`. This covers every VIB-specific tensor.

### A4. EMA state sync client↔server — **CRITICAL (BLOCKING)**
This is the most important bug. Trace:
1. Server's `iterate()` aggregates shared params → `self.model.load_state_dict(global_dict, strict=False)` (feddsa_sgpa.py:441). Since `prototype_ema` is in `private_keys`, it's NOT overwritten here — server retains its own copy. Good.
2. `_update_prototype_ema` then writes the new EMA into `self.model.semantic_head.prototype_ema`. Good.
3. Next round, `Server.pack()` does `copy.deepcopy(self.model)` — deepcopy DOES copy buffers, so the server's updated `prototype_ema` is packed into `svr_pkg['model']`.
4. Client's `unpack()` (feddsa_sgpa.py:586-606) iterates over `local_dict.keys()` and skips `style_head`, BN running stats, and `M` — but **does NOT skip `prototype_ema` / `prototype_init` / `log_var_head` / `log_sigma_prior`**. Result:
   - `prototype_ema` and `prototype_init` ARE synced to client → good (that's what we want).
   - BUT `log_var_head` and `log_sigma_prior` are ALSO overwritten by global model → **BAD**: they become equal to server's stale values (which were never updated because server marks them private and never aggregates). Effectively `log_var_head` parameters stay at init (or near-init) across all rounds; each client's local gradient updates are wiped on every `unpack`.

**Concrete impact**: VIB σ-head never learns. Client-side `log_var_head.parameters()` get optimizer step, but every round resets to server's un-updated copy. KL loss degenerates toward deterministic encoder.

**Fix**: update `Client.unpack` (or extend it in `feddsa_sgpa_vib.Client`) to also skip `log_var_head`, `log_sigma_prior`, and — if you want client-side prototype freedom — `prototype_ema`/`prototype_init` (but for prototype you actually WANT server→client sync, so keep it).

Note: subclass `Server._init_agg_keys` sets `self.private_keys` correctly, but the `Client.unpack` is inherited from `feddsa_sgpa.py:586` which hardcodes its skip list. Subclass must override unpack.

### A5. Server/Client init order — **Minor**
Server.initialize sets `c.vib`, `c.us`, etc. AFTER calling `super().initialize()` (which constructs clients and runs client.initialize). But `Client.initialize()` uses `getattr(self, 'vib', 0)` — during Client.initialize, `c.vib` isn't yet set by Server's explicit loop. However, `Server.init_algo_para` merges `{vib, us, lib, lsc, vws, vwe, sct}` into the dict and flgo's `init_algo_para` propagates attributes to clients via `self._object_map` (fedbase.py:709-710). So `c.vib` etc. ARE set by `flgo.init_algo_para` before Server's explicit loop. The redundant `for c in self.clients: c.vib = ...` loop in Server.initialize is a safe no-op — BUT the naming differs: flgo sets `c.vib` directly, while Server's loop sets `c.vib` too; names match, fine.

However, the Server `initialize()` loop sets `c.lambda_ib = float(getattr(self, 'lib', 1.0))` — here the attribute name changes from `lib` (flgo key) to `lambda_ib`. Client.initialize reads `getattr(self, 'lambda_ib', 1.0)`, so it relies on Server's loop having run first. This works, but is fragile — duplicate naming (`lib` vs `lambda_ib`, `lsc` vs `lambda_supcon`, `vws` vs `vib_warmup_start`, etc.) invites drift. **Minor**.

## Category B: Design Issues

### B6. Loss balance — **Major**
At R≥50, loss is `CE + 1.0*L_orth + 1.0*KL + 1.0*SupCon`. Given smoke kl_mean≈23.7 and CE≈1.5, `λ_IB=1.0` means KL dominates by 10–20×. This is why accuracy regressed smoke round-3→round-5 (0.53→0.46). **Recommendation**: start with `lib=0.01` or `0.001`. Reference: standard VIB (Alemi 2017) uses `β=1e-3` to `1e-2`; practical VIB in vision uses `β=1e-4`.

### B7. Warmup 50 too long — **Minor / possibly OK**
For R200 training, 25% warmup is generous but not unusual. CDANN uses R20→R40 (10% warmup). 50-round warmup means at R50 you suddenly inject full λ_IB=1 — if accompanied by step-function KL=~20+, this will shock training. Combine with B6 fix (reduce λ_IB).

### B8. SupCon in FL small batch — **Major**
PACS batch=50, 7 classes ≈ 7 samples/class, but with non-IID per client, class distribution per batch is skewed. `supcon_loss` correctly skips anchors with 0 positives, but when only 1–2 classes appear in a batch, the loss reduces to near-zero informational content (near-all samples are positives or near-all negatives). More worrying: `valid_mask.sum()` can be very low in some iterations → noisy gradient. On Office-Caltech10 with 10 classes and batch=50, expect ~5 samples/class best case, 0–1 worst. **Recommendation**: log `supcon_n_positive_avg` in diagnostics (already done in `supcon_diagnostics`) and alert if <3.

### B9. mu_sem vs stochastic z_sem inconsistency — **Minor (correct choice)**
The code uses stochastic `z_sem` for CE (task loss) and deterministic `mu_sem` for L_orth and SupCon. This is intentional and correct: stochastic for regularization objective (CE must see noise for VIB), deterministic for auxiliary objectives (avoid double-stochasticity). Class centers upload also uses mu_sem (stability). OK.

Subtle concern: `model.forward()` at inference uses `get_semantic(h)` which calls `semantic_head(h, y=None, training=self.training)`. At eval, training=False → returns mu. Good. **But**: `model.training` is set by caller; if caller forgets `model.eval()`, inference will be stochastic. flgo's test path calls `model.eval()`, should be fine.

### B10. Prototype reach to client VIBSemanticHead — See A4
(Handled above. Once A4 fix applied, prototype reaches client correctly via deepcopy(model).)

## Category C: Experimental Comparison

### C11. Seed alignment with FDSE baseline — **None (OK)**
Seeds 2/15/333 are set via `option['seed']`, flgo uses this to seed data split and torch RNG deterministically. Batch iteration via flgo's `get_batch_data` is seeded. Data splits will match FDSE baselines.

### C12. Config-level parity with baselines — **Major (watch)**
Comparing `feddsa_vib_pacs_r200.yml` vs `feddsa_baseline_pacs_saveckpt_r200.yml`:

| param | baseline | vib | match? |
|---|---|---|---|
| lo | 1.0 | 1.0 | ✅ |
| ue | 0 | 0 | ✅ |
| uw | 1 | 1 | ✅ |
| uc | 1 | 1 | ✅ |
| lr | 0.05 | 0.05 | ✅ |
| E | 5 | 5 | ✅ |
| R | 200 | 200 | ✅ |
| ca | 0 | 0 | ✅ |

Non-VIB hyperparams identical. 

### C13. Office `uc=1` vs EXP-110 `uc=0` — **Critical for fair comparison (BLOCKING)**
EXP-110 Office orth_only ran with `uc=0` (no center aggregation). Your new Office VIB/VSC/SupCon configs all use `uc=1`. This is a confound: if VIB beats EXP-110 baseline 89.09, you cannot attribute the gain to VIB alone — uc=1 adds center-based diagnostics and (importantly) class centers are used for Server's `_update_prototype_ema`. For the SupCon-only variant (vib=0), `uc=1` has zero algorithmic effect beyond diagnostic logging, but for VIB variants the prototype needs centers, so uc=1 is required.

**Recommendation**: also rerun Office orth_only with `uc=1` as a same-config baseline (or confirm EXP-110 uc=0 vs uc=1 diff is negligible; note EXP-110 notes say uc did not affect PACS much). Otherwise, main-table Office comparison will have a reviewer-flag footnote.

### C14. Missing L_aug / L_HSIC / L_InfoNCE — **Minor / acknowledged**
The VIB client has an explicit comment: parent feddsa_sgpa doesn't wire L_aug, so we skip it. L_InfoNCE is explicitly replaced by either `0` (vib=1, us=0 — only KL pulls toward prototype) or SupCon (us=1). This is intentional per round-3 proposal — no `MISSING_CAPABILITY` but document clearly. For variant A (vib=1 us=0), the only prototype-pulling signal is KL. If KL is too weak (λ_IB small, per B6), the semantic head has almost no global anchor → may drift. Consider: for A, either keep λ_IB at 1.0 with other fixes, or re-add a weak InfoNCE as safety net.

## Category D: Deployment Readiness

### D15. Silent-failure modes
| Mode | Detection metric |
|---|---|
| σ collapse (log_var → -5) | `sigma_sem_mean` < 0.1 |
| KL explode | `kl_mean` > 200 or NaN loss |
| prototype never reaches client | `kl_mean ≈ 0` on client side (A4 bug) |
| SupCon no positives | `supcon_n_positive_avg` < 1 |
| CE overwhelmed by KL | task loss plateau + low accuracy |
| Grad cross-cancel (CE vs KL) | cos_sim(grad_CE, grad_KL) near zero — not currently logged |

Add alerts on: `sigma_sem_mean` drift, `kl_mean` explosion (>100), `n_positive_avg<2` on SupCon variants.

### D16. Config deployment risks
- Config lists 20 algo_para values. flgo parses by position. Any off-by-one silently mis-assigns: e.g. swapping `us` and `vib` would flip variants. **Recommendation**: at Server.initialize end, log the resolved `(vib, us, lib, lsc, vws, vwe, sct)` values to the diag log — one-liner, catches off-by-one.
- `log_file: True`, `no_log_console: True`, `local_test: True`: inherited pattern, no issues.

### D17. Record JSON filename overflow — **Major (BLOCKING risk)**
flgo writes records as `{algo}_{algo_para_joined}_M{model}_R{R}_B{B}_E{E}_LR_P_S_LD_WD_SIM_LG.json`. With 20 algo_para values, filename base is ~180 chars + full prefix ≈ 250+ bytes. Linux ext4 NAME_MAX=255.

Sample: `feddsa_sgpa_vib_lo1.0_te0.1_pd128_wr10_es0.001_mcw2_dg1_ue0_uw1_uc1_se1_lp0_ca0_vib1_us0_lib1.0_lsc1.0_vws20_vwe50_sct0.07_Mdefault_model_R200_B50.0_E5_LR5.00e-02_P1.00e+00_S2_LD0_9.9980e-01_WD1.0000e-03_SIMSimulator_LGPerRunLogger.json` → count this: ≈ 265 chars. **Will fail to create/save on ext4**.

**Fix immediately**: flgo `init_algo_para` stores `self.algo_para` as an ordered dict; check how flgo encodes it in filenames. Typical PFLlib/flgo behavior truncates or uses first N chars. Check `FDSE_CVPR25/task/PACS_c4/record/` filename convention on your existing `ditto_*` / FDSE records for exact pattern.

Workaround options:
1. Shorter algo_para keys (already short).
2. Drop unused `sct` / `vws` / `vwe` from algo_para when us=0 or vib=0 — but flgo positional parsing forbids this.
3. Override `Server.algo_para_to_str()` if flgo exposes such a hook.
4. Symlink-based result collection post-hoc.

**Before deploying 18 runs, run ONE R1 smoke on server to verify record JSON actually gets created and path is valid.** The local R5 smoke log in refine-logs only mentions diag; confirm the record file exists.

## Top 3 BLOCKING issues — must fix before deploy

1. **[A4] `Client.unpack` does not skip VIB-private keys.** Override `Client.unpack` in `feddsa_sgpa_vib.Client` to add `log_var_head`, `log_sigma_prior` to the skip list (`prototype_ema`/`prototype_init` should sync server→client, so keep them). Without this fix, the σ-head never learns across rounds. **One-paragraph fix, 15 lines.**

2. **[D17] Filename length overflow on ext4.** Verify by running an R1 smoke and checking `FDSE_CVPR25/task/PACS_c4/record/feddsa_sgpa_vib_*.json` is actually written. If truncated/missing, shorten algo_para keys further or hack a shortened identifier. Discovering this at round 200 costs 18×training time.

3. **[B6 + C13] λ_IB=1.0 likely too large AND Office uc=1 ≠ EXP-110 uc=0 baseline.** Combine: first run a 1-seed R200 PACS with `lib=0.01` to check it's not over-regularized (smoke R5 accuracy 0.16→0.53→0.46 is a yellow flag), and add a parallel Office uc=1 orth_only baseline for fair comparison.

## Top 3 SUGGESTED fixes (non-blocking)

1. **[A1] Clamp `log_sigma_prior` to `[-3, 2]` inside `VIBSemanticHead.forward`** to prevent drift blowup over 200 rounds.

2. **[D16] Print resolved algo_para dict at Server.initialize end**, logged to diag. One print line = foolproof off-by-one defense.

3. **[B8] Log a warning if `supcon_n_positive_avg < 2` for >10 consecutive rounds** — SupCon is degenerating to noise in that regime.

## OVERALL VERDICT: **FIX_BEFORE_DEPLOY**

One critical bug (A4) — σ-head silently fails to learn — would waste 18 GPU runs and invalidate both VIB variants' conclusions. Once A4 is fixed and D17 is smoke-verified on the actual target server (ext4 filesystem), everything else is polish. A4 fix is trivial (~15 lines); D17 is just one-shot R1 verification. B6 (λ_IB) and C13 (Office uc) affect interpretation but not crash-free execution — could be addressed in a follow-up if you want to start the expensive runs quickly, but for A/B verdict they matter.

Recommended pre-deploy checklist:
- [ ] Add `log_var_head`, `log_sigma_prior` to `Client.unpack` skip list (A4)
- [ ] Run R1 smoke on target server, confirm record JSON filename valid (D17)
- [ ] Reduce `lib` to 0.01 in at least one PACS VIB config as hedged run (B6)
- [ ] Add Office orth_only uc=1 baseline config for fair comparison (C13)
- [ ] Log resolved algo_para at Server init end (D16)
