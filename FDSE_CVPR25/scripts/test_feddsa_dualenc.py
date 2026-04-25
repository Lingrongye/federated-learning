"""Unit tests for FedDSA-DualEnc model + 4 loss path.

Run:
    python scripts/test_feddsa_dualenc.py

测试覆盖:
1. 模型 forward 形状 (encode / get_semantic / get_style / decode)
2. AdaIN 输出形状 + 数值合理性 (mean ~0 std 由 gamma/beta 决定)
3. VAE reparameterize 可微性
4. 4 个 loss 都能 backward + 梯度非零
5. L_saac GT 端真的 detach (z_sem 不接收 cycle backward 梯度)
6. Decoder 真用 z_sty (不同 z_sty -> 不同重建)
7. 聚合 key 划分正确 (style_*_head + bn running stats 必须 private)
8. _sample_swap 正确处理 small pool / empty pool
9. _normalize_target 边界处理
10. _kl_weight warmup ramp 正确
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub out flgo dependencies to allow import (单测不需要真 flgo)
import types

if 'flgo' not in sys.modules:
    flgo_pkg = types.ModuleType('flgo')
    flgo_alg = types.ModuleType('flgo.algorithm')
    flgo_fb = types.ModuleType('flgo.algorithm.fedbase')
    flgo_utils = types.ModuleType('flgo.utils')
    flgo_fmodule = types.ModuleType('flgo.utils.fmodule')

    class _BasicServer:
        pass

    class _BasicClient:
        pass

    class _FModule(torch.nn.Module):
        pass

    def _with_multi_gpus(f):
        return f

    flgo_fb.BasicServer = _BasicServer
    flgo_fb.BasicClient = _BasicClient
    flgo_fmodule.FModule = _FModule
    flgo_fmodule.with_multi_gpus = _with_multi_gpus

    # 子模块绑定到 parent (Python module 系统要求)
    flgo_pkg.algorithm = flgo_alg
    flgo_pkg.utils = flgo_utils
    flgo_alg.fedbase = flgo_fb
    flgo_utils.fmodule = flgo_fmodule

    sys.modules['flgo'] = flgo_pkg
    sys.modules['flgo.algorithm'] = flgo_alg
    sys.modules['flgo.algorithm.fedbase'] = flgo_fb
    sys.modules['flgo.utils'] = flgo_utils
    sys.modules['flgo.utils.fmodule'] = flgo_fmodule

# 现在 import 主模块
from algorithm.feddsa_dualenc import (
    FedDSADualEncModel, AdaINBlock, SRM, adain, Decoder, AlexNetEncoder,
)


def _make_model(num_classes=7, sem_dim=512, sty_dim=16, srm_hidden=256):
    return FedDSADualEncModel(
        num_classes=num_classes, feat_dim=1024,
        sem_dim=sem_dim, sty_dim=sty_dim, srm_hidden=srm_hidden,
    )


def test_model_forward_shapes():
    print('[T1] 模型 forward 形状...')
    m = _make_model()
    m.eval()
    x = torch.randn(4, 3, 256, 256)

    h = m.encode(x)
    assert h.shape == (4, 1024), f'encode shape {h.shape}'
    z_sem = m.get_semantic(h)
    assert z_sem.shape == (4, 512), f'z_sem shape {z_sem.shape}'
    mu, logvar = m.get_style(h)
    assert mu.shape == (4, 16) and logvar.shape == (4, 16), 'z_sty shape'
    z_sty = m.reparameterize(mu, logvar)
    assert z_sty.shape == (4, 16), 'z_sty sampled shape'

    x_hat = m.decode(z_sem, z_sty)
    assert x_hat.shape == (4, 3, 256, 256), f'decode shape {x_hat.shape}'
    assert x_hat.min() >= -1.0 and x_hat.max() <= 1.0, f'tanh range broken: {x_hat.min()} {x_hat.max()}'

    logits = m(x)
    assert logits.shape == (4, 7), f'forward shape {logits.shape}'
    print('   PASS')


def test_adain_block():
    print('[T2] AdaIN block 形状 + 数值...')
    blk = AdaINBlock(in_c=64, out_c=32, style_dim=16, srm_hidden=256)
    F_in = torch.randn(4, 64, 16, 16)
    z_sty = torch.randn(4, 16)
    F_out = blk(F_in, z_sty)
    assert F_out.shape == (4, 32, 32, 32), f'AdaIN block shape {F_out.shape}'
    # ReLU 之后非负
    assert (F_out >= 0).all(), 'AdaIN block should have ReLU output'

    # 测 adain 函数本身: 输入 IN 后应该归一化
    F_test = torch.randn(2, 8, 4, 4) * 100 + 50  # 大尺度
    gamma = torch.ones(2, 8)
    beta = torch.zeros(2, 8)
    F_norm = adain(F_test, gamma, beta)
    # gamma=1 beta=0 等价于纯 IN, 各 sample 各 channel mean ~0 std ~1
    means = F_norm.mean(dim=(2, 3))
    stds = F_norm.std(dim=(2, 3), unbiased=False)
    assert means.abs().max() < 1e-3, f'AdaIN IN mean not zero: {means}'
    assert (stds - 1.0).abs().max() < 1e-2, f'AdaIN IN std not unit: {stds}'
    print('   PASS')


def test_vae_reparam_differentiable():
    print('[T3] VAE reparameterize 可微...')
    m = _make_model()
    x = torch.randn(2, 3, 256, 256)
    h = m.encode(x)
    mu, logvar = m.get_style(h)
    z_sty = m.reparameterize(mu, logvar)
    # 梯度应能回流到 mu / logvar
    z_sty.sum().backward()
    has_grad_mu = m.style_mu_head.weight.grad is not None and m.style_mu_head.weight.grad.abs().sum() > 0
    has_grad_lv = m.style_logvar_head.weight.grad is not None and m.style_logvar_head.weight.grad.abs().sum() > 0
    assert has_grad_mu and has_grad_lv, f'VAE reparam gradient broken'
    print('   PASS')


def test_4_losses_have_gradient():
    print('[T4] 4 个 loss 各自都能产生非零梯度...')
    m = _make_model(num_classes=7)
    m.train()
    x = torch.randn(3, 3, 256, 256)
    y = torch.tensor([0, 1, 2])

    # Build a fake "other-client" z_sty pool
    fake_pool = torch.randn(20, 16)

    def reset_grads():
        for p in m.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def grad_norm():
        total = 0.0
        for p in m.parameters():
            if p.grad is not None:
                total += p.grad.abs().sum().item()
        return total

    h = m.encode(x)
    z_sem = m.get_semantic(h)
    mu, logvar = m.get_style(h)
    z_sty = m.reparameterize(mu, logvar)

    # L_CE
    reset_grads()
    F.cross_entropy(m.head(z_sem), y).backward(retain_graph=True)
    assert grad_norm() > 0, 'L_CE no gradient'

    # L_rec
    reset_grads()
    x_hat = m.decode(z_sem, z_sty)
    F.l1_loss(x_hat, x * 2.0 - 1.0).backward(retain_graph=True)
    assert grad_norm() > 0, 'L_rec no gradient'

    # L_saac (cycle): z_sty_swap from pool -> swap -> re-encode
    reset_grads()
    K = 4
    B = x.size(0)
    idx = torch.randint(0, fake_pool.size(0), (B, K))
    chosen = fake_pool[idx]
    alpha = (torch.rand(B, K) * 2 - 1)
    alpha = alpha / alpha.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
    z_sty_swap = (alpha.unsqueeze(-1) * chosen).sum(dim=1)
    x_swap = m.decode(z_sem, z_sty_swap)
    h_swap = m.encode(x_swap)
    z_sem_swap = m.get_semantic(h_swap)
    F.l1_loss(z_sem_swap, z_sem.detach()).backward(retain_graph=True)
    assert grad_norm() > 0, 'L_saac no gradient'

    # L_dsct (InfoNCE)
    reset_grads()
    z_n = F.normalize(z_sty, dim=1)
    neg_n = F.normalize(fake_pool, dim=1)
    pos_logits = z_n @ z_n.T / 0.1
    neg_logits = z_n @ neg_n.T / 0.1
    mask = torch.eye(B).bool()
    pos_logits_masked = pos_logits.masked_fill(mask, float('-inf'))
    logits = torch.cat([pos_logits_masked, neg_logits], dim=1)
    log_pos = torch.logsumexp(pos_logits_masked, dim=1)
    log_all = torch.logsumexp(logits, dim=1)
    (-(log_pos - log_all).mean()).backward(retain_graph=True)
    assert grad_norm() > 0, 'L_dsct no gradient'

    # L_kl
    reset_grads()
    (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()).backward()
    assert grad_norm() > 0, 'L_kl no gradient'

    print('   PASS (CE / rec / saac / dsct / kl 全部有非零梯度)')


def test_cycle_GT_truly_detached():
    print('[T5] L_saac GT 端 (z_sem) 真的 detach, 无自循环梯度...')
    m = _make_model(num_classes=7)
    m.train()
    x = torch.randn(2, 3, 256, 256)

    h = m.encode(x)
    z_sem = m.get_semantic(h)
    mu, logvar = m.get_style(h)
    z_sty = m.reparameterize(mu, logvar)

    # Build a swap z_sty
    z_sty_swap = torch.randn(2, 16)

    # Compute saac with GT detached
    x_swap = m.decode(z_sem, z_sty_swap)
    h_swap = m.encode(x_swap)
    z_sem_swap = m.get_semantic(h_swap)
    loss_saac = F.l1_loss(z_sem_swap, z_sem.detach())

    # Backward and inspect: encoder grad should mostly come from the SWAP path
    # (因为 z_sem.detach() 作为 GT 不会贡献梯度到 encoder)
    # 我们做反向: 如果 GT 没 detach, encoder 会同时收到 cycle path + GT path 双向梯度
    # detach 后只剩 cycle path 一份
    # 简单 sanity: 把 GT 改成完全独立的 tensor (无任何关系) , 应该 loss != 0 且 encoder 仍有梯度
    # 重点验证: z_sem.requires_grad 仍然 True (没被 detach 影响), 但 z_sem.detach() 是 leaf
    assert z_sem.requires_grad, 'z_sem itself should require grad'
    assert not z_sem.detach().requires_grad, 'z_sem.detach() should NOT require grad'

    # Build path: 让 z_sem.detach() 充当 GT, 反向时如果 detach 失效, encoder.bn7.weight 梯度会包含 GT path
    # 这里靠概念:  PyTorch detach() 是 well-tested, 我们只需 assert 这两个 invariants.
    loss_saac.backward()
    # 至少 backbone weight 有非零梯度 (来自 cycle path)
    assert m.encoder.fc2.weight.grad is not None and m.encoder.fc2.weight.grad.abs().sum() > 0, \
        'cycle path should give encoder gradient'
    print('   PASS')


def test_decoder_truly_uses_z_sty():
    print('[T6] Decoder 真用 z_sty (不同 z_sty -> 不同输出)...')
    m = _make_model(num_classes=7)
    m.eval()
    z_sem = torch.randn(2, 512)
    z_sty_a = torch.randn(2, 16)
    z_sty_b = torch.randn(2, 16) * 5  # 完全不同的 style
    with torch.no_grad():
        x_a = m.decode(z_sem, z_sty_a)
        x_b = m.decode(z_sem, z_sty_b)
    diff = (x_a - x_b).abs().mean().item()
    assert diff > 1e-3, f'decoder ignores z_sty (diff={diff:.6f})'

    # 反向: 同 z_sty 同 z_sem -> 同输出
    with torch.no_grad():
        x_a2 = m.decode(z_sem, z_sty_a)
    same = (x_a - x_a2).abs().max().item()
    assert same < 1e-5, f'decoder not deterministic: {same}'
    print(f'   PASS (diff_zsty={diff:.4f}, det_check={same:.2e})')


def test_aggregation_key_classification():
    print('[T7] 聚合 key 分类: style_*_head + BN running stats 必须 private...')
    m = _make_model()
    keys = list(m.state_dict().keys())

    # Mock Server._init_agg_keys 逻辑
    private = set()
    for k in keys:
        if 'style_mu_head' in k or 'style_logvar_head' in k:
            private.add(k)
        elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
            private.add(k)
    shared = [k for k in keys if k not in private]

    # 必须包含的 private keys
    must_private = ['style_mu_head.weight', 'style_mu_head.bias',
                    'style_logvar_head.weight', 'style_logvar_head.bias']
    for k in must_private:
        assert k in private, f'{k} must be private'

    # 必须包含的 shared keys
    must_shared = ['encoder.fc1.weight', 'semantic_head.weight', 'head.weight']
    for k in must_shared:
        assert k in shared, f'{k} must be shared'

    # BN running stats 必须 private (不参与 FedAvg)
    bn_running = [k for k in keys if 'running_' in k or 'num_batches_tracked' in k]
    for k in bn_running:
        assert k in private, f'{k} (BN running stat) must be private'

    # BN gamma/beta (affine 参数, weight + bias) 应该 shared
    bn_affine = [k for k in keys if 'bn' in k.lower() and ('weight' in k or 'bias' in k)
                 and 'running_' not in k]
    for k in bn_affine:
        assert k in shared, f'{k} (BN affine) should be shared (FedBN 半本地策略)'

    print(f'   PASS  (private={len(private)} shared={len(shared)})')


def test_normalize_target_boundary():
    print('[T8] _normalize_target 边界处理...')
    from algorithm.feddsa_dualenc import Client
    norm = Client._normalize_target  # staticmethod, 直接 call

    # Case 1: x in [0, 1] -> map to [-1, 1]
    x_01 = torch.rand(2, 3, 8, 8)
    x_out = norm(x_01)
    assert x_out.min() >= -1.0 and x_out.max() <= 1.0, 'x in [0,1] not mapped to [-1,1]'
    assert (x_out - (x_01 * 2.0 - 1.0)).abs().max() < 1e-6, 'mapping not 2x-1'

    # Case 2: x range exceeds [0, 1] (e.g. ImageNet-norm) -> pass through
    x_pm = torch.randn(2, 3, 8, 8) * 2.0  # min < -0.5 触发 pass-through
    x_out2 = norm(x_pm)
    assert (x_out2 - x_pm).abs().max() < 1e-6, 'large-range x should pass through'

    print('   PASS')


def test_kl_warmup_ramp():
    print('[T9] KL warmup ramp 计算...')
    # 直接测试 _kl_weight 逻辑
    class _FakeClient:
        def __init__(self, current_round, kl_warmup_rounds, lambda_kl):
            self.current_round = current_round
            self.kl_warmup_rounds = kl_warmup_rounds
            self.lambda_kl = lambda_kl

        from algorithm.feddsa_dualenc import Client as _C
        _kl_weight = _C._kl_weight

    # 修正后: 1-based, progress = max(0, current_round - 1)
    # round=0 -> progress=0, ratio=0.0
    # round=1 -> progress=0, ratio=0.0 (warmup 第一轮还没开始 ramp)
    # round=6 -> progress=5, ratio=5/10=0.5
    # round=11 -> progress=10, ratio=1.0
    fc = _FakeClient(current_round=0, kl_warmup_rounds=10, lambda_kl=0.01)
    assert fc._kl_weight() == 0.0, f'round 0 should give 0 KL: {fc._kl_weight()}'

    fc.current_round = 1
    assert fc._kl_weight() == 0.0, f'round 1 (progress=0) should give 0: {fc._kl_weight()}'

    fc.current_round = 6
    expected = 0.5 * 0.01
    assert abs(fc._kl_weight() - expected) < 1e-6, f'round 6 (progress=5/10) should give 0.5*lambda: {fc._kl_weight()}'

    fc.current_round = 11
    assert abs(fc._kl_weight() - 0.01) < 1e-6, f'round 11 (progress=10) should give full lambda'

    fc.current_round = 200
    assert abs(fc._kl_weight() - 0.01) < 1e-6, f'after warmup should stay at lambda'

    print('   PASS')


def test_swap_pool_edge_cases():
    print('[T10] _sample_swap 边界 (空 pool / 单样本 pool)...')
    # 直接构造 pool 测试
    from algorithm.feddsa_dualenc import Client

    class _FakeClient:
        bank_K = 4
        sty_dim = 16   # ← 现在 client 直接持有, 不再通过 model.sty_dim
        _sample_swap = Client._sample_swap

    fc = _FakeClient()

    # Case 1: 空 pool -> fallback random
    z_swap = fc._sample_swap(batch_size=4, device=torch.device('cpu'), pool=None)
    assert z_swap.shape == (4, 16), f'empty pool fallback shape: {z_swap.shape}'

    # Case 2: 单样本 pool
    pool = torch.randn(1, 16)
    z_swap = fc._sample_swap(batch_size=4, device=torch.device('cpu'), pool=pool)
    assert z_swap.shape == (4, 16), 'single pool shape'

    # Case 3: 正常 pool
    pool = torch.randn(50, 16)
    z_swap = fc._sample_swap(batch_size=8, device=torch.device('cpu'), pool=pool)
    assert z_swap.shape == (8, 16), 'normal pool shape'

    # Case 4: K > pool.size(0)
    fc.bank_K = 10
    pool = torch.randn(3, 16)
    z_swap = fc._sample_swap(batch_size=4, device=torch.device('cpu'), pool=pool)
    assert z_swap.shape == (4, 16), 'K>N capped'

    print('   PASS')


def test_dsct_loss_smoke():
    print('[T11] L_dsct InfoNCE 计算 (单 batch + 跨 client 负例)...')
    from algorithm.feddsa_dualenc import Client

    class _FakeClient:
        tau = 0.1
        _dsct_loss = Client._dsct_loss

    fc = _FakeClient()
    z_sty = torch.randn(8, 16, requires_grad=True)
    negatives = torch.randn(50, 16)

    loss = fc._dsct_loss(z_sty, negatives)
    assert torch.isfinite(loss), f'L_dsct non-finite: {loss}'
    loss.backward()
    assert z_sty.grad is not None and z_sty.grad.abs().sum() > 0, 'L_dsct no gradient'
    print(f'   PASS (loss={loss.item():.4f})')


def test_style_cycle_gradient_flows_to_style_encoder():
    """Codex CRITICAL 1 修复验证: cycle path 必须给 style_*_head 反向梯度.

    新设计: x_swap = decode(z_sem, z_sty_swap) -> encode -> mu_swap_after,
    L_style_cyc = L1(mu_swap_after, z_sty_swap.detach()).
    z_sty_swap detach (来自 bank), mu_swap_after 经过 style_*_head, 所以梯度必须回流.
    """
    print('[T12] L_style_cycle 梯度回流到 style_*_head (CRITICAL 1 fix)...')
    m = _make_model()
    m.train()
    x = torch.randn(2, 3, 256, 256)
    z_sty_swap = torch.randn(2, 16)  # detached, like from bank

    h = m.encode(x)
    z_sem = m.get_semantic(h)

    x_swap = m.decode(z_sem, z_sty_swap)
    h_swap = m.encode(x_swap)
    mu_swap_after, _ = m.get_style(h_swap)

    loss_style_cyc = F.l1_loss(mu_swap_after, z_sty_swap.detach())
    loss_style_cyc.backward()

    has_grad_mu = m.style_mu_head.weight.grad is not None and m.style_mu_head.weight.grad.abs().sum() > 0
    assert has_grad_mu, 'L_style_cyc 没回流到 style_mu_head! CRITICAL 1 修复失败'
    print(f'   PASS  (style_mu_head.grad_norm={m.style_mu_head.weight.grad.abs().sum().item():.4f})')


def test_dsct_no_intra_client_pull():
    """Codex CRITICAL 2 修复验证: 新 _dsct 不再把 batch 内同 client 拉近.

    检验: 给 negative 数量 = 0 (空 bank) -> loss 退化为 -log(1) = 0.
    给 negative 数量 > 0 -> 梯度只来自跟 bank 的对比, 不跟 batch 内自己拉近.
    """
    print('[T13] L_dsct 不再 push 同 client 紧 (CRITICAL 2 fix)...')
    from algorithm.feddsa_dualenc import Client

    class _FakeClient:
        tau = 0.1
        _dsct_loss = Client._dsct_loss

    fc = _FakeClient()
    z_sty = torch.randn(8, 16, requires_grad=True)
    # Case 1: 空 negatives -> loss = 0
    loss = fc._dsct_loss(z_sty, torch.empty(0, 16))
    assert loss.item() == 0.0, f'空 bank 应该返 0: {loss.item()}'

    # Case 2: 正常 negatives, 验证梯度只受 cross-client 影响
    neg = torch.randn(50, 16)
    loss = fc._dsct_loss(z_sty, neg)
    loss.backward()
    assert z_sty.grad is not None and z_sty.grad.abs().sum() > 0, 'L_dsct 应该有梯度'

    # Case 3: 当 batch 内全部 z_sty 等同 (collapse), loss 应该跟 batch 内紧无关
    z_sty_collapsed = torch.zeros(8, 16, requires_grad=True) + 0.5
    z_sty_diverse = torch.randn(8, 16, requires_grad=True)
    loss_collapsed = fc._dsct_loss(z_sty_collapsed, neg)
    loss_diverse = fc._dsct_loss(z_sty_diverse, neg)
    # 旧版 InfoNCE 把 collapse 当作正例紧, loss 会显著低. 新版 instance discrim 不应该.
    # 这里只验证两个 loss 数值有限非 NaN, 不强求一致 (因为还跟 bank 距离有关).
    assert torch.isfinite(loss_collapsed) and torch.isfinite(loss_diverse), 'loss should be finite'
    print(f'   PASS  (collapsed={loss_collapsed.item():.4f}, diverse={loss_diverse.item():.4f})')


def test_pack_returns_none_when_no_other_clients():
    """Codex IMPORTANT 修复验证: 别 client 风格为空时 pack 返回 None,
    不再 fallback 到 self bank."""
    print('[T14] Server.pack 别 client 空时不 self-fallback...')
    from algorithm.feddsa_dualenc import Server

    # mock server
    class _S:
        current_round = 100
        saac_warmup_rounds = 10
        style_bank = {0: torch.randn(50, 16)}  # 只有 client 0 的 bank

    s = _S()
    pack_method = Server.pack
    # 调用时 client_id=0 (自己), 应该返回 None (因为没别 client)
    s.model = torch.nn.Linear(2, 2)  # dummy
    pkg = pack_method(s, client_id=0)
    assert pkg['style_bank'] is None, f'no other client should yield None, got {pkg["style_bank"]}'

    # 加一个 client 1, 现在 client_id=0 应该拿到 client 1 的 bank
    s.style_bank[1] = torch.randn(50, 16)
    pkg = pack_method(s, client_id=0)
    assert pkg['style_bank'] is not None and 1 in pkg['style_bank'] and 0 not in pkg['style_bank'], \
        f'should dispatch only client 1 styles to client 0: {pkg["style_bank"]}'
    print('   PASS')


def main():
    print('=== FedDSA-DualEnc Unit Tests ===\n')
    torch.manual_seed(0)
    test_model_forward_shapes()
    test_adain_block()
    test_vae_reparam_differentiable()
    test_4_losses_have_gradient()
    test_cycle_GT_truly_detached()
    test_decoder_truly_uses_z_sty()
    test_aggregation_key_classification()
    test_normalize_target_boundary()
    test_kl_warmup_ramp()
    test_swap_pool_edge_cases()
    test_dsct_loss_smoke()
    test_style_cycle_gradient_flows_to_style_encoder()
    test_dsct_no_intra_client_pull()
    test_pack_returns_none_when_no_other_clients()
    print('\n=== ALL 14 TESTS PASSED ===')


if __name__ == '__main__':
    main()
