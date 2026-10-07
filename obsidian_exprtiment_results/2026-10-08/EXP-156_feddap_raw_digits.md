# EXP-156：FedDAP 原始四域 Digits smoke

状态：准备中，尚未启动训练。

计划从公开来源重新下载 MNIST / USPS / SVHN / SYN；所有原始文件校验完成后才允许启动训练，不能用既有裁剪缓存替代。

- 配置：`experiments/sanity/EXP-156_feddap_raw_digits/config.json`。
- 本地诊断：`experiments/sanity/EXP-156_feddap_raw_digits/diag_exp156_digits_s2_raw4_r2_e1_v1/`。
- 远端诊断：`/home/lry/code/feddap-exp155/experiments/sanity/EXP-156_feddap_raw_digits/diag_exp156_digits_s2_raw4_r2_e1_v1/`。
- 原始源文件：`/home/lry/data/feddap_digits_raw/`。

这次只验收原始数据读取和 FedDAP 的真实训练闭环，不代表论文主表准确率已经复现。
