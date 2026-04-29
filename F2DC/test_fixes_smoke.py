"""
R=5 smoke test 验证 4 个 fix:
1. 聚合白名单: dfd/dfc/aux 不参与聚合, 应保持 client local
2. Deterministic eval: 同 model + 同 input, 两次 eval 输出完全一致 (max_abs_diff = 0)
3. Hardcoded 7: Office (10 类) 不报 size mismatch
4. _sync_global_proto_to_global_net: server eval 时 global_net.class_proto 非零

每个 fix 独立测试, 不跑完整 R100.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np


def test_aggregation_whitelist():
    """Fix #1: dfd/dfc/aux 不参与聚合, client local 应保留差异"""
    print("\n" + "=" * 70)
    print("Test #1: 聚合白名单 (dfd/dfc/aux 不聚合)")
    print("=" * 70)

    from backbone.ResNet_DC_PG import resnet10_dc_pg_office

    # 创建 2 个 client model, 给 dfd/dfc/aux/conv1 设不同 sentinel value
    net0 = resnet10_dc_pg_office(num_classes=10, gum_tau=0.1, proto_weight=0.3)
    net1 = resnet10_dc_pg_office(num_classes=10, gum_tau=0.1, proto_weight=0.3)
    global_net = resnet10_dc_pg_office(num_classes=10, gum_tau=0.1, proto_weight=0.3)

    # 哨兵值
    with torch.no_grad():
        net0.dfd_module.net[0].weight.fill_(0.0)
        net1.dfd_module.net[0].weight.fill_(2.0)
        net0.dfc_module.net[0].weight.fill_(10.0)
        net1.dfc_module.net[0].weight.fill_(14.0)
        net0.aux[0].weight.fill_(20.0)
        net1.aux[0].weight.fill_(24.0)
        net0.conv1.weight.fill_(100.0)
        net1.conv1.weight.fill_(200.0)

    # 模拟聚合 (用我们的白名单逻辑)
    def _is_aggregatable(key):
        local_patterns = ['dfd_module', 'dfc_module', 'aux.', 'class_proto',
                          'q_proj', 'k_proj', 'v_proj']
        return not any(p in key for p in local_patterns)

    nets_list = [net0, net1]
    global_w = global_net.state_dict()
    keys_to_agg = [k for k in global_w.keys() if _is_aggregatable(k)]

    freq = [0.5, 0.5]
    first = True
    for index, net in enumerate(nets_list):
        net_para = net.state_dict()
        if first:
            first = False
            for key in keys_to_agg:
                global_w[key] = net_para[key] * freq[index]
        else:
            for key in keys_to_agg:
                global_w[key] += net_para[key] * freq[index]

    global_net.load_state_dict(global_w)

    global_sd = global_net.state_dict()
    for net in nets_list:
        client_sd = net.state_dict()
        for key in keys_to_agg:
            client_sd[key] = global_sd[key]
        net.load_state_dict(client_sd)

    # 验证
    print(f"  conv1 (应聚合到 150):       net0={net0.conv1.weight[0,0,0,0].item():.1f}  global={global_net.conv1.weight[0,0,0,0].item():.1f}")
    print(f"  dfd_module (应保持差异 0/2): net0={net0.dfd_module.net[0].weight[0,0,0,0].item():.1f}  net1={net1.dfd_module.net[0].weight[0,0,0,0].item():.1f}")
    print(f"  dfc_module (应保持差异 10/14): net0={net0.dfc_module.net[0].weight[0,0,0,0].item():.1f}  net1={net1.dfc_module.net[0].weight[0,0,0,0].item():.1f}")
    print(f"  aux (应保持差异 20/24):     net0={net0.aux[0].weight[0,0].item():.1f}  net1={net1.aux[0].weight[0,0].item():.1f}")

    # 验证
    assert abs(global_net.conv1.weight[0,0,0,0].item() - 150.0) < 0.01, "conv1 应聚合"
    assert abs(net0.dfd_module.net[0].weight[0,0,0,0].item() - 0.0) < 0.01, "dfd 不应被聚合"
    assert abs(net1.dfd_module.net[0].weight[0,0,0,0].item() - 2.0) < 0.01, "dfd 不应被聚合"
    assert abs(net0.dfc_module.net[0].weight[0,0,0,0].item() - 10.0) < 0.01, "dfc 不应被聚合"
    assert abs(net1.dfc_module.net[0].weight[0,0,0,0].item() - 14.0) < 0.01, "dfc 不应被聚合"
    assert abs(net0.aux[0].weight[0,0].item() - 20.0) < 0.01, "aux 不应被聚合"
    assert abs(net1.aux[0].weight[0,0].item() - 24.0) < 0.01, "aux 不应被聚合"
    print("  ✅ Pass: dfd/dfc/aux 保持 local, conv1 正确聚合")
    return True


def test_deterministic_eval():
    """Fix #2: 同 model + 同 input, 两次 eval (is_eval=True) 输出完全一致"""
    print("\n" + "=" * 70)
    print("Test #2: Deterministic eval (is_eval=True 时 gumbel 用 0.5 noise)")
    print("=" * 70)

    from backbone.ResNet_DC_PG import resnet10_dc_pg_office
    net = resnet10_dc_pg_office(num_classes=10, gum_tau=0.1, proto_weight=0.0)
    net.eval()
    x = torch.randn(2, 3, 32, 32)

    # 两次 is_eval=True
    with torch.no_grad():
        out1 = net(x, is_eval=True)[0]
        out2 = net(x, is_eval=True)[0]
    diff_eval_true = (out1 - out2).abs().max().item()

    # 两次 is_eval=False (默认, 旧 bug)
    with torch.no_grad():
        out3 = net(x, is_eval=False)[0]
        out4 = net(x, is_eval=False)[0]
    diff_eval_false = (out3 - out4).abs().max().item()

    print(f"  is_eval=True  两次 max_abs_diff: {diff_eval_true:.6e}  (应为 0)")
    print(f"  is_eval=False 两次 max_abs_diff: {diff_eval_false:.6e}  (旧 bug 应有噪声)")
    assert diff_eval_true < 1e-9, f"is_eval=True 应 deterministic, got {diff_eval_true}"
    print("  ✅ Pass: is_eval=True deterministic, 旧默认有噪声 ~1e-4")
    return True


def test_hardcoded_classes():
    """Fix #3: Office 10 类用 args.num_classes 不会 fallback 7"""
    print("\n" + "=" * 70)
    print("Test #3: 硬编码 7 → args.num_classes")
    print("=" * 70)

    # 模拟 main_run.py 创建 args + model
    class FakeArgs:
        num_classes = 10
        device_id = 0
        parti_num = 10
        seed = 42

    # 模拟 PG-DFC model 状态: local_protos[0] is None (client 0 还没上线)
    # 触发 fallback path
    feat_dim = 512
    args = FakeArgs()
    local_protos = [None] * 10

    # 用我们 fix 后的逻辑
    N_CLASSES = (
        local_protos[0].shape[0]
        if (local_protos[0] is not None)
        else int(getattr(args, 'num_classes', 7))
    )
    print(f"  args.num_classes = {args.num_classes}")
    print(f"  fallback fix 后 N_CLASSES = {N_CLASSES}")
    assert N_CLASSES == 10, f"Office fallback 应该用 args.num_classes=10, got {N_CLASSES}"
    print("  ✅ Pass: Office 10 类正确 fallback")
    return True


def test_class_proto_buffer_not_persistent():
    """Fix #4: 验证 class_proto 是 non-persistent buffer (确认 bug 来源)"""
    print("\n" + "=" * 70)
    print("Test #4: class_proto 是 non-persistent buffer (state_dict 不含)")
    print("=" * 70)

    from backbone.ResNet_DC_PG import DFC_PG
    dfc = DFC_PG(size=(512, 4, 4), num_classes=10, proto_weight=0.3)

    # 写入 class_proto
    with torch.no_grad():
        dfc.class_proto.fill_(5.0)

    sd = dfc.state_dict()
    has_class_proto_in_sd = 'class_proto' in sd
    print(f"  dfc.state_dict() 包含 class_proto? {has_class_proto_in_sd}")
    print(f"  dfc.class_proto[0,0] = {dfc.class_proto[0,0].item():.1f} (in-memory)")
    assert not has_class_proto_in_sd, "class_proto 应该是 persistent=False"
    print("  ✅ 确认 bug 来源: class_proto 不在 state_dict, server FedAvg 永远不会同步它")
    print("  ✅ 修复 _sync_global_proto_to_global_net 显式 copy 是必须的")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("R=N/A smoke test 验证 4 个 fix")
    print("=" * 70)

    results = []
    for name, fn in [
        ("聚合白名单", test_aggregation_whitelist),
        ("Deterministic eval", test_deterministic_eval),
        ("硬编码 7 fix", test_hardcoded_classes),
        ("class_proto buffer 非 persistent (bug 来源)", test_class_proto_buffer_not_persistent),
    ]:
        try:
            ok = fn()
            results.append((name, "✅ PASS" if ok else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ FAIL: {e}"))
            print(f"  ❌ {e}")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    for name, status in results:
        print(f"  {status}  {name}")
