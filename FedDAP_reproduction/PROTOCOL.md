# FedDAP 复现边界与实现协议

## 来源
- 论文：FedDAP, arXiv:2604.06795v1（包含补充材料），CVPR 2026。
- 官方代码：quanghuy6997/FedDAP_CVPR2026，
  本地上游 revision `988211eb826d81251bd8b3b8ae53fbdcc7f3dc20`。
- `run_digits.py` 是独立修复入口，不经过官方损坏的模块注册表。
  原仓库及用户原有中文注释不修改；这不是声称原 main.py 全部修好了。
- 使用已跟踪的 `F2DC/backbone/ResNet.py`；其可执行 AST 与上述上游
  `backbone/ResNet.py` 相同，仅注释与空白不同，不引入 F2DC 算法。

## 论文算法必须保留的链路
1. 同一全局状态广播给全部客户端；本地 SGD，CE + λ1 DPA + λ2 CPCL。
2. DPA 对齐同域同类原型；CPCL 正例为其他域同类，负例为其他域异类。
   排除所有本域原型，参考原型 detach；logsumexp 与公式的指数比值等价。
3. 本地训练结束后，用最终编码器重新遍历完整已分配数据，计算类均值。
4. 服务器仅融合相同 `(class, domain)`，余弦相似度行求和排除对角线，
   温度 softmax 后加权求和；单个原型的权重为 1。
5. 模型按客户端真实样本数进行 FedAvg，含浮点 BN 缓冲；全局分类器评估。
6. 不引入 LAB、FedBN、私有分类器、额外解耦模块或未定义的 DP 噪声。

## 论文明确的正式实验设置（不是本次短程设置）
- DomainNet / Office-10 / PACS；100 轮，每轮本地 10 epoch，全部客户端参与。
- ResNet10 / ResNet10 / ResNet18；batch 32 / 32 / 16。
- DomainNet 域客户端数 3/1/4/6/4/2，
  Office 为 caltech3/amazon2/webcam1/dslr4，
  PACS 为 photo3/art2/cartoon1/sketch4。
- 客户端采样比例分别 10% / 20% / 30%。
- 三次重复，使用最后五轮的平均；不能用 best accuracy 替代正式指标。

## 未明确或存在冲突的细节：不得假装已完全对齐
- 正文 Eq.6 的 DPA 求和与公开代码的 batch mean 不完全明确；
  当前保留公开代码 batch mean，缺失锚点贡献 0，计入全 batch 分母。
- 论文没有明确最终编码器原型遍历的 BN mode 与增强策略；
  当前使用 eval BN + 确定性测试变换，不污染 BN，覆盖全部本地样本。
- 首轮无原型、仅 CE：采用公开实现的冷启动约定。
- BN 整数计数器取 max；浮点 BN 参数/统计量参与 FedAvg。
- 公开代码把未定义 `dp()` 引入默认路径；主实验不施加噪声。
  若复现补充材料隐私实验，必须另行定义并核验噪声协议。
- PACS 同一域 4 客户端 × 每客户端 30% 超过 100%，
  与公开代码无放回分配冲突；重叠采样还是比例定义必须澄清。
- 主表精确 λ/温度、重复 seed、数据 split/图像尺度等信息仍需配置或作者确认。
  超参分析最优点不能擅自当作所有主表的已知配置。

## 本次 Digits smoke 的显式偏离
- 只用服务器现有真实 MNIST / MNIST_M / SVHN 缓存，
  不下载、伪造或称为完整 Digit-5；USPS / SynthDigits 不在这个缓存里。
- 每域 3 客户端；每客户端无重叠抽 128 训练样本，每域抽 128 测试样本。
- seed2，2 轮 × 1 local epoch，scratch ResNet10，batch32，lr0.01，
  SGD momentum0.9 / weight_decay1e-5，
  λ1=λ2=1，τcross=0.02，τagg=0.001（公开 Digits 默认及聚合代码）。
- 32×32 与 ImageNet normalize、随机裁剪/水平翻转沿用公开 Digits 变换。
- 这些不是论文主表设置，只验证算法/梯度/原型/聚合/保存的真实链路。

## 运行和数据留存
在项目根目录用 pfllib Python 执行：
`CUBLAS_WORKSPACE_CONFIG=:4096:8 python FedDAP_reproduction/run_digits.py --data-root <trusted-cache-root> --dump-diag <unique-directory> --device cuda:0`
原始 pkl 是已有可信 NumPy 对象缓存；不接受未知来源对象文件。
诊断目录与 npz 文件均独占创建，重跑必须新路径。
共享 GPU 上单模型顺序训练，本进程 torch allocator 上限为显卡总内存的 10%，
启动时可用显存少于 4GiB 则失败退出；不终止、不改变其他人的任务。
保留每轮 npz、best/final（模型与真实测试特征）、meta、data_manifest、
proto_logs、summary；任何失败保留 failure.json 及此前产物，完整 rsync 回本地。
`meta.json` 是启动快照，完成状态看 `summary.json`，失败看 `failure.json`。
