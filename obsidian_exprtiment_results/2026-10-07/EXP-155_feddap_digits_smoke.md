# EXP-155 | FedDAP 真实 Digits smoke 在 lab-lry 跑通

- 创建日期：2026-10-07；编号按既有最大EXP-154继续，不修改任何旧实验。
- 状态：真实CUDA运行完成，完整原始诊断回传，服务器与本地验收PASS。
- 代码：FedDAP_reproduction/；保留官方clone及用户原注释不变。
- source revision：e761a05ded977b16a020b93e5cdf5b59b94013b1。
- 完整记录：[NOTE.md](../../experiments/sanity/EXP-155_feddap_digits_smoke/NOTE.md)。

## 做了什么
使用服务器现有MNIST / MNIST-M / SVHN真实缓存，scratch ResNet10，
seed2、2轮、每轮1local_epoch、每域3客户端，每客户端128训练图，
域内无重叠；每域128个固定测试样本。只验证链路，不是完整Digit-5，
也不是FedDAP正式论文数据集。

算法保留CE+DPA+CPCL、训练结束后最终编码器完整原型重算、
同域同类余弦注意力融合、样本数加权FedAvg、全局分类器推理。
没有LAB、FedBN、私有分类器或未定义的DP噪声。

## 怎样看结果
域准确率=正确预测数÷128×100%，范围0–100%，高表示预测好；
AVG是3域准确率等权平均。这里只有seed2的2轮短测，没有效果达标阈值，
不与FDSE或论文数字比较；只以有限损失/梯度、真实链路及产物可还原判PASS。

R2域正确数11/13/22，对应MNIST8.59375%、MNIST-M10.15625%、SVHN17.1875%，
AVG11.97917%。没有收敛或效果证明，也没有完整末五轮；summary中paper_result_reproduced=false。

关键验收是R2所有客户端DPA/CPCL有效锚点和独立特征梯度非零；
每轮30个`(class,domain)`键覆盖本次10类×3域；所有best/final严格还原模型并
用对应真实特征重算逐域预测成功。所有原始文件回传SHA256逐字一致。

## 原始诊断位置（不得覆盖或删除）
- 本地：experiments/sanity/EXP-155_feddap_digits_smoke/diag_exp155_digits_s2_r2_e1_gpu1_v2/
- 服务器：/home/lry/code/feddap-exp155/experiments/sanity/EXP-155_feddap_digits_smoke/diag_exp155_digits_s2_r2_e1_gpu1_v2/
- round_001/002.npz、best_R001.npz、final_R002.npz、meta.json、proto_logs.jsonl、
  data_manifest.json、attempt.json、summary.json全部保留；日志在run_gpu1_v2/。
- 本次heavy各约39MB，不超单文件限制，全部入Git；无删旧数据或排除round文件。

## 还不能称为严格完整复现
新入口仅有Digits smoke。正式论文需要Office/PACS/DomainNet等协议和正式末五轮、多次重复；
DPA归一化、原型遍历模式、精确超参、PACS分配冲突仍需澄清，
见[FedDAP协议](../../FedDAP_reproduction/PROTOCOL.md)。
Codex CLI独立审查因账号模型兼容失败，采用工具独立只读审查并修复其P1、
补测试；该偏离在VALIDATION.md明文记录，不虚构CLI成功。
