# EXP-155 | FedDAP Digits 真实数据 smoke test

## 基本信息
- 创建日期：2026-10-07（Asia/Shanghai）。
- 类型：sanity；状态：准备中，尚未启动训练。
- 方法：FedDAP；目标服务器：lab-lry。
- 上游源码：FedDAP_CVPR2026，起始 revision 988211eb826d81251bd8b3b8ae53fbdcc7f3dc20。

## 目的与证据边界
通过独立复现入口绕开官方实现的运行阻塞，验证真实 Digits 数据的本地训练、最终编码器原型重算、
域内对齐 DPA、跨域对比 CPCL、同域同类原型聚合、全局参数聚合和全局分类器评估链路。
Digits 不是 FedDAP 论文正式实验数据集；本实验不能用于宣称论文准确率已复现。
不引入 LAB、双头解耦、FedBN 或其他新方法；不存在的 LAB 字段不伪造。

## 计划
- 先核查论文正文及补充材料，记录明确设置与未交代设置。
- 测试损失公式、梯度、聚合、真实数据接口及诊断保留。
- 原上游仓库及用户注释保持不变；新入口位于 FedDAP_reproduction/。
- 沿用与上游可执行 AST 一致的已跟踪 ResNet 文件，单测验证一致性。
- Git 同步后再在 lab-lry 启动。
- 至少两个通信轮次，保证后一轮实际使用首轮上传的原型。
- 独立 dump_diag 目录，拒绝复用已有目录；完成或失败均完整回传。

## 诊断路径
- 本地：本目录/diag_exp155_digits_s2_r2_e1_v1/。
- 服务器：/home/lry/code/feddap-exp155/experiments/sanity/EXP-155_feddap_digits_smoke/diag_exp155_digits_s2_r2_e1_v1/。
- 参数如有变动必须另用新路径，不覆盖旧数据。
- 必须保留 round_*.npz、best_R*.npz、final_R*.npz、meta.json、proto_logs.jsonl。
- 保存模型、原型、样本索引、逐域准确率和特征；heavy snapshot 本地保留。

## 指标说明
准确率 = 域内正确预测数 / 该域测试样本数 × 100%，范围 0–100%，越高越好。
AVG = 各测试域准确率的等权平均，不按域样本数加权。
本 smoke test 只检查链路、有限损失/梯度、原型覆盖和保存完整性，不设论文准确率门槛。
短程测试没有完整末五轮，不将其汇总冒充论文主表结果。

## 运行与结果
待核查服务器环境与数据后填写。没有已验证的训练结果。

## 已核查的环境、数据与配置
- lab-lry pfllib：Python3.11，torch2.6.0+cu124，torchvision0.21.0+cu124，
  numpy2.4.3，Pillow12.1.1；不重建、不升级共享环境。
- 现有可信缓存：FedPLVM/data/digit/digit/{MNIST,MNIST_M,SVHN}。
  该目录无 USPS/SynthDigits；本次明确只用三个真实域，不声称完整 Digit-5。
- 每域3客户端，每客户端128训练样本，域内不重叠；每域128测试样本。
- seed2，2轮×1local_epoch，batch32，lr0.01，λDPA=λCPCL=1，
  τcross0.02，τagg0.001；详细参数保存在不可覆盖的 config.json。
- GPU 两卡都存在其他任务；短程低显存单模型顺序执行，不终止他人进程。
- 正式论文协议与未明确细节：FedDAP_reproduction/PROTOCOL.md。
