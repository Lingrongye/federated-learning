"""
完整覆盖测试 f2dc_pg_ml — 8 个独立 test case 验证:

T1. backbone forward shape: 7-tuple 接口 + aux3 shape + mask3 shape
T2. transient attribute: _last_aux3_logits 在多 batch 下 refresh, 不持久
T3. eval determinism: is_eval=True 两次 forward 主输出一致 + aux3 一致
T4. state_dict roundtrip: dfd_lite/dfc_lite/aux3 在 state_dict, transient 不在
T5. FedAvg 兼容: 两个 net 全 state_dict 平均后 load 回来不报 size mismatch
T6. gradient flow: alpha=0.1 时 dfd_lite/dfc_lite/aux3 都有梯度; alpha=0 时全 None
T7. alpha=0 退化等价: 跟 PG-DFC v3.3 forward 主路输出 bit-identical
T8. PACS / Office shape 兼容: 两套 image_size 都不报 shape error

不依赖真实数据集, 全部用 torch.randn 构造数据.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import copy


def banner(title):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def assert_close(a, b, atol=1e-6, msg=''):
    diff = (a - b).abs().max().item()
    assert diff < atol, f'{msg} max_abs_diff={diff:.6e} > {atol:.6e}'


def test_T1_forward_shape():
    """T1: forward 返回 7-tuple, aux3/mask3 shape 正确"""
    banner('T1: forward shape (7-tuple + aux3/mask3)')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office, resnet10_dc_pg_ml
    # Office (32×32 input)
    net = resnet10_dc_pg_ml_office(num_classes=10)
    net.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ret = net(x, is_eval=True)
    assert isinstance(ret, tuple) and len(ret) == 7, f'expected 7-tuple, got {len(ret)}'
    out, feat, ro_outputs, re_outputs, rec_outputs, ro_flat, re_flat = ret
    assert out.shape == (2, 10), f'out {out.shape}'
    assert feat.shape == (2, 512), f'feat {feat.shape}'
    assert net._last_aux3_logits.shape == (2, 10), f'aux3 {net._last_aux3_logits.shape}'
    assert net._last_mask3.shape[0] == 2 and net._last_mask3.shape[1] == 256, f'mask3 {net._last_mask3.shape}'
    print(f'  Office: out {out.shape} aux3 {net._last_aux3_logits.shape} mask3 {tuple(net._last_mask3.shape)}')

    # PACS (128×128 input)
    net2 = resnet10_dc_pg_ml(num_classes=7)
    net2.eval()
    x2 = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out2, feat2, _, _, _, _, _ = net2(x2, is_eval=True)
    assert out2.shape == (2, 7) and feat2.shape == (2, 512)
    assert net2._last_aux3_logits.shape == (2, 7)
    assert net2._last_mask3.shape == (2, 256, 32, 32)
    print(f'  PACS:   out {out2.shape} aux3 {net2._last_aux3_logits.shape} mask3 {tuple(net2._last_mask3.shape)}')
    print('  ✅ Pass')


def test_T2_transient_attribute_refresh():
    """T2: _last_aux3_logits 每次 forward 刷新, 不被前次 forward 污染"""
    banner('T2: transient attribute refresh')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office
    net = resnet10_dc_pg_ml_office(num_classes=10)
    net.eval()
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        net(x1, is_eval=True)
        a1 = net._last_aux3_logits.clone()
        net(x2, is_eval=True)
        a2 = net._last_aux3_logits.clone()
    diff = (a1 - a2).abs().max().item()
    assert diff > 1e-6, f'_last_aux3_logits 没刷新, max_diff={diff}'
    print(f'  两次不同输入 aux3 max_abs_diff = {diff:.4f} (应 > 0)')
    print('  ✅ Pass')


def test_T3_eval_determinism():
    """T3: is_eval=True 两次 forward 主输出一致 + aux3 一致"""
    banner('T3: eval determinism (is_eval=True)')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office
    torch.manual_seed(0)
    net = resnet10_dc_pg_ml_office(num_classes=10)
    net.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out1, _, _, _, _, _, _ = net(x, is_eval=True)
        a1 = net._last_aux3_logits.clone()
        out2, _, _, _, _, _, _ = net(x, is_eval=True)
        a2 = net._last_aux3_logits.clone()
    assert_close(out1, out2, atol=1e-9, msg='主输出不 deterministic')
    assert_close(a1, a2, atol=1e-9, msg='aux3 不 deterministic')
    print(f'  主输出 max_abs_diff = {(out1-out2).abs().max().item():.2e} (应 = 0)')
    print(f'  aux3   max_abs_diff = {(a1-a2).abs().max().item():.2e} (应 = 0)')
    # is_eval=False 应该有 random 噪声
    with torch.no_grad():
        out3, _, _, _, _, _, _ = net(x, is_eval=False)
        out4, _, _, _, _, _, _ = net(x, is_eval=False)
    diff_random = (out3 - out4).abs().max().item()
    assert diff_random > 0, '随机模式应有差异'
    print(f'  is_eval=False max_abs_diff = {diff_random:.2e} (应 > 0)')
    print('  ✅ Pass')


def test_T4_state_dict_roundtrip():
    """T4: dfd_lite/dfc_lite/aux3 在 state_dict, transient (_last_*) 不在"""
    banner('T4: state_dict roundtrip')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office
    net = resnet10_dc_pg_ml_office(num_classes=10)
    sd = net.state_dict()
    keys = list(sd.keys())
    has_dfd_lite = any(k.startswith('dfd_lite.') for k in keys)
    has_dfc_lite = any(k.startswith('dfc_lite.') for k in keys)
    has_aux3 = any(k.startswith('aux3.') for k in keys)
    no_transient = not any(k.startswith('_last_') for k in keys)
    has_class_proto = 'dfc_module.class_proto' in keys
    assert has_dfd_lite, '缺 dfd_lite.* in state_dict'
    assert has_dfc_lite, '缺 dfc_lite.* in state_dict'
    assert has_aux3, '缺 aux3.* in state_dict'
    assert no_transient, '_last_* 不应在 state_dict'
    # class_proto 是 persistent=False, 应该不在 state_dict
    assert not has_class_proto, 'class_proto 不应在 state_dict (persistent=False)'
    print(f'  dfd_lite keys: {[k for k in keys if k.startswith("dfd_lite.")][:3]}')
    print(f'  dfc_lite keys: {[k for k in keys if k.startswith("dfc_lite.")][:3]}')
    print(f'  aux3 keys: {[k for k in keys if k.startswith("aux3.")]}')
    print(f'  无 _last_*: {no_transient}, 无 class_proto: {not has_class_proto}')
    # roundtrip
    net2 = resnet10_dc_pg_ml_office(num_classes=10)
    net2.load_state_dict(sd)
    sd2 = net2.state_dict()
    for k in keys:
        assert (sd[k] == sd2[k]).all(), f'{k} 不一致 after roundtrip'
    print('  state_dict roundtrip 全部 key 一致')
    print('  ✅ Pass')


def test_T5_fedavg_compat():
    """T5: 两个 net 全 state_dict FedAvg 平均后 load 回来不报错"""
    banner('T5: FedAvg compatibility (整 state_dict 平均)')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office
    net0 = resnet10_dc_pg_ml_office(num_classes=10)
    net1 = resnet10_dc_pg_ml_office(num_classes=10)
    global_net = resnet10_dc_pg_ml_office(num_classes=10)
    # 模拟 federated_model.aggregate_nets 的全 state_dict 平均
    nets_list = [net0, net1]
    global_w = global_net.state_dict()
    freq = [0.5, 0.5]
    first = True
    for index, net in enumerate(nets_list):
        net_para = net.state_dict()
        if first:
            first = False
            for key in net_para:
                global_w[key] = net_para[key] * freq[index]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * freq[index]
    global_net.load_state_dict(global_w)
    # 各 client load global state
    for net in nets_list:
        net.load_state_dict(global_net.state_dict())
    # 重新 forward 不崩
    global_net.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out, _, _, _, _, _, _ = global_net(x, is_eval=True)
    assert out.shape == (2, 10)
    print(f'  FedAvg + load_state_dict 后 global forward shape={out.shape}')
    print('  ✅ Pass')


def test_T6_gradient_flow():
    """T6: alpha=0.1 时 dfd_lite/dfc_lite/aux3 都有梯度; alpha=0 时全 None"""
    banner('T6: gradient flow (lite branch)')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office
    criterion = nn.CrossEntropyLoss()

    # Case A: alpha=0.1 (full deep sup)
    net = resnet10_dc_pg_ml_office(num_classes=10)
    net.train()
    x = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    out, _, ro_outputs, _, _, _, _ = net(x)
    aux3_logits = net._last_aux3_logits
    main_loss = criterion(out, labels)
    aux3_loss = criterion(aux3_logits, labels)
    total = main_loss + 0.1 * aux3_loss
    total.backward()
    g_dfd_lite = net.dfd_lite.net[0].weight.grad
    g_dfc_lite = net.dfc_lite.net[0].weight.grad
    g_aux3 = net.aux3.weight.grad
    g_layer4 = net.layer4[0].conv1.weight.grad
    assert g_dfd_lite is not None and g_dfd_lite.abs().sum() > 0, 'dfd_lite 无梯度'
    assert g_dfc_lite is not None and g_dfc_lite.abs().sum() > 0, 'dfc_lite 无梯度'
    assert g_aux3 is not None and g_aux3.abs().sum() > 0, 'aux3 无梯度'
    assert g_layer4 is not None and g_layer4.abs().sum() > 0, 'layer4 无梯度'
    print(f'  alpha=0.1: dfd_lite |g|={g_dfd_lite.abs().sum().item():.4e} '
          f'aux3 |g|={g_aux3.abs().sum().item():.4e}')

    # Case B: alpha=0 (lite 不参与 loss)
    net2 = resnet10_dc_pg_ml_office(num_classes=10)
    net2.train()
    out2, _, _, _, _, _, _ = net2(x)
    main_loss2 = criterion(out2, labels)
    main_loss2.backward()  # 不加 aux3_loss
    g_dfd_lite2 = net2.dfd_lite.net[0].weight.grad
    g_aux32 = net2.aux3.weight.grad
    g_layer4_2 = net2.layer4[0].conv1.weight.grad
    # alpha=0 时 lite 模块没参与 loss → grad 应为 None
    assert g_dfd_lite2 is None or g_dfd_lite2.abs().sum() == 0, \
        f'alpha=0 时 dfd_lite 不应有梯度, got |g|={g_dfd_lite2.abs().sum() if g_dfd_lite2 is not None else "None"}'
    assert g_aux32 is None or g_aux32.abs().sum() == 0, \
        f'alpha=0 时 aux3 不应有梯度, got |g|={g_aux32.abs().sum() if g_aux32 is not None else "None"}'
    assert g_layer4_2 is not None and g_layer4_2.abs().sum() > 0, \
        'alpha=0 时 layer4 仍应有梯度 (主路 backward)'
    g_dfd_lite2_str = 'None' if g_dfd_lite2 is None else f'{g_dfd_lite2.abs().sum().item():.4e}'
    g_aux32_str = 'None' if g_aux32 is None else f'{g_aux32.abs().sum().item():.4e}'
    print(f'  alpha=0:   dfd_lite |g|={g_dfd_lite2_str} aux3 |g|={g_aux32_str} '
          f'layer4 |g|={g_layer4_2.abs().sum().item():.4e} (应 > 0)')
    print('  ✅ Pass')


def test_T7_alpha0_equivalence():
    """T7: alpha=0 时主路 forward 输出跟 PG-DFC v3.3 完全一致 (相同 seed/init)"""
    banner('T7: alpha=0 → 主路输出等价 PG-DFC v3.3')
    from backbone.ResNet_DC_PG_ML import resnet10_dc_pg_ml_office
    from backbone.ResNet_DC_PG import resnet10_dc_pg_office

    torch.manual_seed(42)
    net_ml = resnet10_dc_pg_ml_office(num_classes=10, gum_tau=0.1, proto_weight=0.0)
    torch.manual_seed(42)
    net_pg = resnet10_dc_pg_office(num_classes=10, gum_tau=0.1, proto_weight=0.0)

    # Copy 共有 module 的参数 — net_pg 的参数应跟 net_ml 的同名参数一致
    # 因为两个都用 manual_seed=42 做 init, 但 net_ml 多了 dfd_lite/dfc_lite/aux3
    # 这些额外 module 的 init 会消耗 RNG 状态, 所以共享 module 的 init 不一致
    # 解决: 显式 copy 共有 module 的 state_dict
    sd_ml = net_ml.state_dict()
    sd_pg = net_pg.state_dict()
    # 把 net_pg 的所有共有 key copy 到 net_ml
    for k in sd_pg:
        if k in sd_ml:
            sd_ml[k] = sd_pg[k]
    net_ml.load_state_dict(sd_ml)

    net_ml.eval()
    net_pg.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_ml, _, _, _, _, _, _ = net_ml(x, is_eval=True)
        out_pg, _, _, _, _, _, _ = net_pg(x, is_eval=True)
    diff = (out_ml - out_pg).abs().max().item()
    assert diff < 1e-5, f'主路输出应一致 (alpha=0 退化), got max_abs_diff={diff:.6e}'
    print(f'  PG-DFC vs PG-DFC-ML 主输出 max_abs_diff = {diff:.2e} (应 ≈ 0)')
    print('  ✅ Pass')


def test_T8_pacs_office_shape():
    """T8: PACS 128×128 + Office 32×32 + Digits 32×32 三种 shape 都不崩"""
    banner('T8: PACS / Office / Digits shape 兼容')
    from backbone.ResNet_DC_PG_ML import (
        resnet10_dc_pg_ml, resnet10_dc_pg_ml_office, resnet10_dc_pg_ml_digits
    )
    cases = [
        ('PACS',   resnet10_dc_pg_ml(num_classes=7),         (2, 3, 128, 128), 7,  (256, 32, 32)),
        ('Office', resnet10_dc_pg_ml_office(num_classes=10), (2, 3, 32, 32),   10, (256, 8, 8)),
        ('Digits', resnet10_dc_pg_ml_digits(num_classes=10), (2, 3, 32, 32),   10, (256, 8, 8)),
    ]
    for name, net, x_shape, n_cls, mask3_chw in cases:
        net.eval()
        x = torch.randn(*x_shape)
        with torch.no_grad():
            out, feat, _, _, _, _, _ = net(x, is_eval=True)
        assert out.shape == (x_shape[0], n_cls), f'{name} out {out.shape}'
        assert net._last_aux3_logits.shape == (x_shape[0], n_cls)
        assert net._last_mask3.shape[1:] == mask3_chw, f'{name} mask3 {tuple(net._last_mask3.shape)}'
        print(f'  {name:7}: in {x_shape} out {out.shape} mask3 {tuple(net._last_mask3.shape[1:])}')
    print('  ✅ Pass')


if __name__ == '__main__':
    print('=' * 70)
    print('  f2dc_pg_ml 完整覆盖测试 (8 个 case)')
    print('=' * 70)

    tests = [
        ('T1', test_T1_forward_shape),
        ('T2', test_T2_transient_attribute_refresh),
        ('T3', test_T3_eval_determinism),
        ('T4', test_T4_state_dict_roundtrip),
        ('T5', test_T5_fedavg_compat),
        ('T6', test_T6_gradient_flow),
        ('T7', test_T7_alpha0_equivalence),
        ('T8', test_T8_pacs_office_shape),
    ]
    results = []
    for tid, fn in tests:
        try:
            fn()
            results.append((tid, '✅ PASS'))
        except Exception as e:
            results.append((tid, f'❌ FAIL: {type(e).__name__}: {e}'))
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 70)
    print('  总结')
    print('=' * 70)
    for tid, status in results:
        print(f'  {status}  {tid}')
    n_pass = sum(1 for _, s in results if s.startswith('✅'))
    print(f'\n  {n_pass}/{len(results)} PASS')
    sys.exit(0 if n_pass == len(results) else 1)
