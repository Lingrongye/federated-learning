# EXP-156：FedDAP 原始 Digits 下载与复跑

创建日期：2026-10-08（Asia/Shanghai）。

## 目的与证据边界

下载公开原始 MNIST、USPS、SVHN、SynthDigits，验证文件来源、完整性和标签后，在 lab-lry 重跑 FedDAP 算法链路。不得用 EXP-155 的裁剪缓存冒充本次原始下载。

这是 Digits 工程 smoke test，不是论文主表结果复现；论文参数、官方 Digits 代码参数和 smoke 缩减参数必须分别记录。

## 数据与诊断保留

- 远端源数据计划路径：`/home/lry/data/feddap_digits_raw`。
- 远端实验目录：`/home/lry/code/feddap-exp155/experiments/sanity/EXP-156_feddap_raw_digits`。
- 计划诊断目录名：`diag_exp156_digits_s2_raw4_r2_e1_v1`；启动前检查不存在，失败重跑使用新的后缀。
- 本地诊断路径：`experiments/sanity/EXP-156_feddap_raw_digits/diag_exp156_digits_s2_raw4_r2_e1_v1`。
- 完成或失败后完整同步所有原始 round/best/final NPZ、配置、数据清单、日志；不删除或覆盖旧实验。

## 当前状态

准备中。尚未确认原始数据下载成功，尚未启动本次训练。

## 可复执行顺序

本地验证（本次本地临时环境另需 scipy==1.17.1；服务器已有此版本，不更新共享环境）：

```bash
/tmp/feddap-exp155-test-env/bin/python FedDAP_reproduction/test_core.py
/tmp/feddap-exp155-test-env/bin/python FedDAP_reproduction/test_raw.py
```

代码提交并 push 后，在服务器隔离 checkout pull，确认 HEAD 一致，再执行：

```bash
cd /home/lry/code/feddap-exp155
nice -n 10 /home/lry/conda/envs/pfllib/bin/python FedDAP_reproduction/raw_data.py --root /home/lry/data/feddap_digits_raw --background-log-dir experiments/sanity/EXP-156_feddap_raw_digits/download_v1
/home/lry/conda/envs/pfllib/bin/python FedDAP_reproduction/wait_run.py --directory /home/lry/data/feddap_digits_raw --kind download --timeout 1200
/home/lry/conda/envs/pfllib/bin/python FedDAP_reproduction/inspect_raw.py --root /home/lry/data/feddap_digits_raw --output experiments/sanity/EXP-156_feddap_raw_digits/download_v1
```

只有 `download_manifest.json` 标记 completed 且所有校验一致才能启动：

```bash
nice -n 10 /home/lry/conda/envs/pfllib/bin/python FedDAP_reproduction/launch.py --config experiments/sanity/EXP-156_feddap_raw_digits/config.json --output-dir experiments/sanity/EXP-156_feddap_raw_digits/run_v1
/home/lry/conda/envs/pfllib/bin/python FedDAP_reproduction/wait_run.py --directory experiments/sanity/EXP-156_feddap_raw_digits/diag_exp156_digits_s2_raw4_r2_e1_v1 --timeout 300
/home/lry/conda/envs/pfllib/bin/python FedDAP_reproduction/verify_artifacts.py experiments/sanity/EXP-156_feddap_raw_digits/diag_exp156_digits_s2_raw4_r2_e1_v1
```

任何失败保留原目录；新下载目录或重跑诊断必须改新名称，不覆盖。

以上命令应在 `set -e` 的 shell 顺序执行；完成门槛返回非零后不得继续验收。启动器只返回 PID，绝不是训练完成证明。完成门槛有超时，并优先检查 failure 文件；确认 summary.status=completed 后才可做产物验收。

来源检查额外保存四域逐类预览 `download_v1/source_preview.png` 及包含原始像素和索引的 `source_samples.npz`；人工检查数字和标签是否一致。预览不是训练集抽样划分，不参与训练。

## 原始来源与协议边界

- MNIST 使用 torchvision 指定的 OSSCI 镜像及四个原始 gzip 文件的官方 MD5。
- USPS 使用 LIBSVM 官方两个 bz2 文件及 torchvision MD5；标签减 1，像素按 torchvision 公式从 [-1,1] 转 uint8。
- SVHN 使用 Stanford 原始 train/test MAT 文件及 torchvision MD5；标签 10 映射到数字 0。
- SynthDigits 原作者主页链接的 Google Drive 目前返回 404。因此采用研究库 `domainadaptation/datasets` 的完整 MAT 镜像，不使用 `_small` 文件；固定提交 `91ff02225a73db330883843bc41ee3aca59a19d9`，按 Git LFS 声明的 SHA256 和字节数校验。
- SYN 是公开发布的合成数字域，不是本次自行生成的假数据。官方 FedDAP 只要求 `syn/train`、`syn/val` ImageFolder，没有提供该转换的数据清单；目前不能证明 MAT 镜像和作者未提供的 ImageFolder 完全等价。
- 保留所有完整下载文件；训练仅为 smoke 从原始 train 抽取每客户端 256 条，各域三个不重叠客户端；各域 test 抽取 512 条。选择索引和类计数全部写入诊断。
- seed=2，两轮，每轮一次本地 epoch；这些 smoke 缩减值不是论文主表实验。模型、损失和原型训练流程沿用 EXP-155 已验收实现。
