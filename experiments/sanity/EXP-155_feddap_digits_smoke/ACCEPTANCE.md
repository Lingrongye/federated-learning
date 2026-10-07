# EXP-155 真实数据 smoke 验收证明

## 合约与边界
目标是固定seed、真实缓存、真实CUDA下完成本地CE+DPA+CPCL、
最终编码器原型重算、同域同类注意力聚合、样本加权FedAvg、全局分类器评估、
完整诊断与可还原snapshot；不是达到论文准确率，更不是复现所有实验。
训练revision：`e761a05ded977b16a020b93e5cdf5b59b94013b1`。

## 实际执行的验收
- lab-lry：`verify_artifacts.py <服务器diag目录>` 输出acceptance=PASS。
- 本地完整rsync后：同一verify_artifacts.py输出acceptance=PASS。
- 验证器检查round文件数、逐域正确数÷测试数的准确率、FedAvg样本比例、
  原型有限值、注意力权重和、R1冷启动/R2两损失和梯度、域内训练索引无重叠。
- 每个heavy snapshot都strict=True还原state_dict，用保存的对应真实测试特征
  通过该全局分类器重算预测；正确数与保存字段逐域完全相同。
- 原始诊断文件9个，全部已回传；下面的hash来自服务器验收器输出，
  本地Node crypto对每个回传文件重算SHA256并逐字比对，全部matched=true。
- hash只证明文件逐字一致，不自动证明公式正确；公式与梯度由单测及独立审查另行验证。

## 服务器原始文件SHA256
| 文件 | SHA256 |
|---|---|
| attempt.json | 77789f40ebe537733ba85cad86d5aaaa233f36832a16f48b67a94ec58fd5dff1 |
| best_R001.npz | 10c5fbb33411ba82bff5e5f691277e97fe7dd730bbb8c15c0e6e5094a1d98cff |
| data_manifest.json | 337886699119699c2384b1f224fdbef74f580f0f3479f6d535f9033fb86bc9a8 |
| final_R002.npz | 6945791e02bbcc2dc4a0e1dcd3b90b34cb92c581df84dc4ef7cbfd8095d9ef4a |
| meta.json | 191233e5d01e869e9c9e415ae5c54ddc75bfc143ad2915aee42ba4139880fe98 |
| proto_logs.jsonl | 357b7bed6c9e76a3f4cb9561130fd009f44729cdf70289a261347fc4aad07f99 |
| round_001.npz | ff3f43e0326e7be4028531c79932e3bdf5bc3a9736b98b770963634aabfa8a00 |
| round_002.npz | 8e5b35081a0cd4c8312ebab33f3ff005b1a89eab6e9804497acf17f02e2073f5 |
| summary.json | 4cf590b851e896d2bce7f611d499bcfef36663c4d9d5294687b6aa8d4a7e2db2 |

## 完整路径
- 本地：experiments/sanity/EXP-155_feddap_digits_smoke/diag_exp155_digits_s2_r2_e1_gpu1_v2/
- 服务器：/home/lry/code/feddap-exp155/experiments/sanity/EXP-155_feddap_digits_smoke/diag_exp155_digits_s2_r2_e1_gpu1_v2/
- 日志与实际启动命令：上述实验目录/run_gpu1_v2/{train.log,launch.json}。
