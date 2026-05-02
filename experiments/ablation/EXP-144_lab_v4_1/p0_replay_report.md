# EXP-144 P0 Offline Replay Report
> LAB v4.2 算法验证 (用 1-test_acc 当 val_loss proxy, 仅验权重轨迹合理性, 不证明 acc 效果)

## Coverage (codex guardrail)

- **Datasets covered**: PACS + Office vanilla cold-path runs (8 total)
- **NOT covered in P0**: Digits (no cold-path npz available; deferred to P1 real val_loss)
- **What this proves**: bisection projection numerical stability, ratio bounds, boost-direction semantics on 2 datasets where DaA shows opposite signs (PACS loses, Office wins)
- **What this does NOT prove**: actual accuracy improvement (proxy 1-acc 太粗, 真实信号在 P1 编码后才有)

**Config**: λ=0.15, ratio∈[0.8, 2.0], EMA α=0.3, bisection tol=1e-09, max_iter=64
---
## ⚖️ 总体判定

**P0 PASS**: ✅ YES

**Warnings**:
- ⚠️ Office f2dc s15: domain saturated >95% rounds: max=[np.str_('dslr')], min=[]

---

## 📊 Per-Run Diagnostics

### PACS f2dc s15 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 1.5535])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.91e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.3 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 0.0%
- Per domain (@max): photo:0%, art:0%, cartoon:0%, sketch:0%
- Per domain (@min): photo:0%, art:0%, cartoon:0%, sketch:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: photo:0.003, art:9.938, cartoon:0.201, sketch:0.366
- Mean boost/round: photo:0.0000, art:0.0994, cartoon:0.0020, sketch:0.0037

---

### PACS f2dc s333 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 1.5429])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.86e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.1 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 0.0%
- Per domain (@max): photo:0%, art:0%, cartoon:0%, sketch:0%
- Per domain (@min): photo:0%, art:0%, cartoon:0%, sketch:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: photo:0.135, art:9.517, cartoon:0.303, sketch:0.000
- Mean boost/round: photo:0.0013, art:0.0952, cartoon:0.0030, sketch:0.0000

---

### PACS pgdfc s15 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 1.5050])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.99e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.2 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 0.0%
- Per domain (@max): photo:0%, art:0%, cartoon:0%, sketch:0%
- Per domain (@min): photo:0%, art:0%, cartoon:0%, sketch:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: photo:0.085, art:9.714, cartoon:0.209, sketch:0.186
- Mean boost/round: photo:0.0009, art:0.0971, cartoon:0.0021, sketch:0.0019

---

### PACS pgdfc s333 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 1.5197])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.45e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.1 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 0.0%
- Per domain (@max): photo:0%, art:0%, cartoon:0%, sketch:0%
- Per domain (@min): photo:0%, art:0%, cartoon:0%, sketch:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: photo:0.031, art:9.890, cartoon:0.305, sketch:0.276
- Mean boost/round: photo:0.0003, art:0.0989, cartoon:0.0031, sketch:0.0028

---

### Office f2dc s2 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 2.0000])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.71e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.3 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 3.5%
- Per domain (@max): caltech:0%, amazon:0%, webcam:2%, dslr:12%
- Per domain (@min): caltech:0%, amazon:0%, webcam:0%, dslr:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: caltech:0.000, amazon:0.000, webcam:6.675, dslr:5.600
- Mean boost/round: caltech:0.0000, amazon:0.0000, webcam:0.0668, dslr:0.0560

---

### Office f2dc s15 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 2.0000])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.60e-10 (✅ < 1e-6)
- Iterations: max=32, mean=31.0 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 24.8%
- Per domain (@max): caltech:0%, amazon:0%, webcam:0%, dslr:99%
- Per domain (@min): caltech:0%, amazon:0%, webcam:0%, dslr:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: [np.str_('dslr')]
- Always @min: none

**Boost Distribution**:
- Total boost: caltech:0.000, amazon:0.000, webcam:3.454, dslr:7.337
- Mean boost/round: caltech:0.0000, amazon:0.0000, webcam:0.0345, dslr:0.0734

---

### Office pgdfc s2 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 2.0000])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.86e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.1 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 2.5%
- Per domain (@max): caltech:0%, amazon:0%, webcam:3%, dslr:7%
- Per domain (@min): caltech:0%, amazon:0%, webcam:0%, dslr:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: caltech:0.000, amazon:0.051, webcam:6.615, dslr:5.643
- Mean boost/round: caltech:0.0000, amazon:0.0005, webcam:0.0662, dslr:0.0564

---

### Office pgdfc s15 (R=100, 4 domains)

**① Algorithm Correctness**:
- Ratios in bounds: **100.0000%** (observed: [0.8500, 2.0000])
- NaN: ✅ NO

**② Bisection Stability**:
- Max sum error: 9.79e-10 (✅ < 1e-6)
- Iterations: max=33, mean=31.1 (✅ < 64)
- Non-converged rounds: 0

**③ Clip Rate**:
- Overall: 15.8%
- Per domain (@max): caltech:0%, amazon:0%, webcam:1%, dslr:62%
- Per domain (@min): caltech:0%, amazon:0%, webcam:0%, dslr:0%

**④ Saturation Detection (>95% rounds at boundary)**:
- Always @max: none
- Always @min: none

**Boost Distribution**:
- Total boost: caltech:0.000, amazon:0.000, webcam:5.045, dslr:7.050
- Mean boost/round: caltech:0.0000, amazon:0.0000, webcam:0.0504, dslr:0.0705

---

