# EXP-155 | FedDAP Digits 真实数据 smoke test

## 基本信息
- 创建日期：2026-10-07（Asia/Shanghai）。
- 类型：sanity；状态：真实GPU smoke完成，服务器与本地诊断验收通过。
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
- 本地：本目录/diag_exp155_digits_s2_r2_e1_gpu1_v2/。
- 服务器：/home/lry/code/feddap-exp155/experiments/sanity/EXP-155_feddap_digits_smoke/diag_exp155_digits_s2_r2_e1_gpu1_v2/。
- 启动前发现GPU1计算空闲而GPU0繁忙，改为cuda:1并使用新参数版本路径。
  原v1只是准备配置，从未启动或创建diag目录，不存在删除/覆盖。
- 参数如有变动必须另用新路径，不覆盖旧数据。
- 必须保留 round_*.npz、best_R*.npz、final_R*.npz、meta.json、proto_logs.jsonl。
- 保存模型、原型、样本索引、逐域准确率和特征；heavy snapshot 本地保留。

## 指标说明
准确率 = 域内正确预测数 / 该域测试样本数 × 100%，范围 0–100%，越高越好。
AVG = 各测试域准确率的等权平均，不按域样本数加权。
本 smoke test 只检查链路、有限损失/梯度、原型覆盖和保存完整性，不设论文准确率门槛。
短程测试没有完整末五轮，不将其汇总冒充论文主表结果。

## 运行与结果
服务器真实训练完成；已完整回传并机械核对全部诊断文件SHA256。
- 训练 source revision：e761a05ded977b16a020b93e5cdf5b59b94013b1。
- 服务器进程 PID：226177（只证明启动，不证明完成）。
- 启动：nice -n 10，独立会话，由 launch.py 保存实际命令。
- 服务器日志：本实验目录/run_gpu1_v2/train.log。
- 启动命令记录：本实验目录/run_gpu1_v2/launch.json。
- config.json 已用于此 run，之后不可变；任何重跑必须新配置/新diag路径。

## 结果与解释
- 域准确率 = 该域预测正确数 / 128个固定真实测试样本 ×100%，范围0–100%，
  越高越好；AVG是3个域准确率等权平均，不按训练样本量加权。
- 这里只跑seed2的2轮×1local_epoch，不是收敛实验，没有论文准确率达标阈值；
  必须以链路/有限梯度/产物可还原验收，不能比较FDSE或论文正式主表。

| 轮次 | MNIST | MNIST-M | SVHN | AVG |
|---|---:|---:|---:|---:|
| R1 | 8.59375% | 7.81250% | 19.53125% | 11.97917% |
| R2 | 8.59375% | 10.15625% | 17.18750% | 11.97917% |

- R2各域正确数为11/13/22，每域分母128；准确率低只说明短程模型尚不能作为效果证据。
- R1没有全局原型，DPA/CPCL为0，符合明确记录的公开实现冷启动约定。
- R2的9个客户端，两项损失均有128/128个有效锚点；两项独立特征梯度均非零。
- 全局原型数是字典 `(class,domain)` 的键数，本次10类×3域=30；
  两轮都覆盖30个键，表示本次类别/域覆盖齐全，不代表准确率好。
- 用summary的单调时钟差计算运行耗时10.450秒，含数据读取、训练、评估和快照，
  不含本地编码、Git同步和服务器前置检查；它不是正式实验的耗时预测。
- torch allocator峰值=231124480字节÷1024²=220.417MiB，仅本进程张量分配，
  不含CUDA上下文或其他用户任务；不能当作nvidia-smi整卡占用。
- 本实验没有完整末五轮，summary.last5_mean=null，paper_result_reproduced=false。

## 留存与验收
- 服务器及本地原始诊断目录均保留；未覆盖、未删除任何已有实验。
- 包含round_001/002.npz、best_R001.npz、final_R002.npz、meta.json、
  proto_logs.jsonl、data_manifest.json、attempt.json、summary.json。
- best与final保存完整全局state_dict及对应384个真实测试特征、标签、域，
  还保存所有客户端本地原型。两个AVG相同，因此只保存首次best，不重复造best_R002。
- verify_artifacts.py在服务器与回传后的本地分别PASS，严格加载各snapshot并用
  保存特征通过对应全局分类器重新计算，逐域正确数与npz完全一致。
- 9个诊断文件SHA256机械比对服务器输出与本地回传一致，详见ACCEPTANCE.md。
- 实际train.log与launch.json完整回传至本目录/run_gpu1_v2/。
- 两个heavy npz各约39MB，均低于GitHub单文件限制，本次全部入Git，不添加忽略规则。

## 尚未复现的部分
新入口目前只支持这次Digits smoke；没有启动Office/PACS/DomainNet正式主表、
消融、未见域泛化、隐私噪声或多seed实验。严格完整复现还必须澄清
PROTOCOL.md中DPA reduction、原型遍历模式、精确超参和PACS分配冲突等细节，
不能由“smoke PASS”推断论文所有结果已复现。

## 已核查的环境、数据与配置
- lab-lry pfllib：Python3.11，torch2.6.0+cu124，torchvision0.21.0+cu124，
  numpy2.4.3，Pillow12.1.1；不重建、不升级共享环境。
- 现有可信缓存：FedPLVM/data/digit/digit/{MNIST,MNIST_M,SVHN}。
  该目录无 USPS/SynthDigits；本次明确只用三个真实域，不声称完整 Digit-5。
- 每域3客户端，每客户端128训练样本，域内不重叠；每域128测试样本。
- seed2，2轮×1local_epoch，batch32，lr0.01，λDPA=λCPCL=1，
  τcross0.02，τagg0.001；详细参数保存在不可覆盖的 config.json。
- GPU 两卡都存在其他任务；短程低显存单模型顺序执行，不终止他人进程。
- 启动前最新检查GPU1利用率0%、可用显存足够；选择cuda:1，训练器另有显存门槛。
- 独立精简Git checkout：/home/lry/code/feddap-exp155；不改变共享原目录及其修改。
- 正式论文协议与未明确细节：FedDAP_reproduction/PROTOCOL.md。
