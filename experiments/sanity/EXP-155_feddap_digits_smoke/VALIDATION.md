# EXP-155 启动前验证记录

## 本地受控验证
- 日期：2026-10-07。
- 独立临时环境：/tmp/feddap-exp155-test-env/bin/python，Python3.12。
- torch2.6.0、torchvision0.21.0、numpy2.4.3、Pillow12.1.1。
- 实际执行：`python FedDAP_reproduction/test_core.py`。
- AST 解析通过；最终版本 13 个单元测试全部 PASS（新增 CUDA 不可用失败留存）。
- 覆盖：单/多正例 CPCL 公式、独立 DPA/CPCL 特征梯度、
  参考原型 detach、同域过滤、极小温度有限值、缺失锚点、
  三原型注意力已知权重、域隔离、样本加权 FedAvg、BN计数器、
  最终编码器完整原型重算、ResNet 前向与梯度、
  缓存适配器三通道且单次变换、npz不可覆盖、初始化失败保留且不能覆盖重跑。
- backbone 可执行 AST 与固定上游一致的测试实际 PASS（不是凭文件名推断）。
- `run_digits.py --help` 实际执行，参数与 config 键映射一致。
- 上游 `git ls-remote origin HEAD` 本次核查仍为
  988211eb826d81251bd8b3b8ae53fbdcc7f3dc20。

## 独立审查与偏离记录
- 仓库规定的 `codex exec -s read-only ...` 已尝试，但 CLI 报当前默认
  gpt-6.1-sol 不受该 ChatGPT 账号的 Codex CLI 支持；**CLI 审查未完成**。
- 替代为工具支持的独立只读审查，不修改模型设置、不修改共享环境。
- 首轮审查发现 Important：预留目录后的初始化错误没有 failure.json。
- 已修复：独占预留与 attempt 后，所有环境/模型/元数据初始化均纳入外层异常保护；
  新增 mock 初始化失败与同路径重跑不可覆盖测试，实际 PASS。
- 替代审查不等同于 compute-env-contract 的“新 agent 逐字运行完整文档命令”
  验收，也不把静态审查当成服务器训练完成。
- 复查还指出 CUDA 可用性检查应进入同一异常保护，已移动并补测试。
- 最终独立复查关闭该 P1；限定 smoke 实现未发现其他 Important+。
- 审查只读判断与本地测试分别记录，不把任一个冒充服务器运行证明。

## 服务器验证
待 Git 同步后执行同一测试及 preflight。真实 GPU smoke 与所有诊断回传另行记录。
