"""
Client for DSA (Decouple-Share-Align) — Decoupled Prototype Learning with Style Asset Sharing
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from flcore.clients.clientbase import Client


class clientDSA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Auto-infer feat_dim from the model head, fallback to CLI arg
        if hasattr(args, 'head') and hasattr(args.head, 'in_features'):
            feat_dim = args.head.in_features
        else:
            feat_dim = args.feat_dim if hasattr(args, 'feat_dim') else 512
        proj_dim = args.proj_dim if hasattr(args, 'proj_dim') else 128
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim

        # Decoupling heads
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ).to(self.device)
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ).to(self.device)

        # Semantic classifier on top of z_sem (FIX: task gradient flows through semantic_head)
        self.sem_classifier = nn.Linear(proj_dim, args.num_classes).to(self.device)

        # Hyperparameters (FIX: lambda_orth and lambda_hsic are independent weights)
        self.lambda_orth = args.lambda_orth if hasattr(args, 'lambda_orth') else 1.0
        self.lambda_hsic = args.lambda_hsic if hasattr(args, 'lambda_hsic') else 0.1
        self.lambda_sem = args.lambda_sem if hasattr(args, 'lambda_sem') else 1.0
        self.tau = args.tau if hasattr(args, 'tau') else 0.1
        self.num_classes = args.num_classes

        # Style augmentation
        self.style_bank = None
        self.global_semantic_protos = None
        self.warmup_rounds = args.warmup_rounds if hasattr(args, 'warmup_rounds') else 10
        self.current_round = 0

        # Optimizer includes all trainable components
        self.optimizer = torch.optim.SGD(
            list(self.model.parameters()) +
            list(self.semantic_head.parameters()) +
            list(self.style_head.parameters()) +
            list(self.sem_classifier.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=1e-5
        )
        # FIX: Rebuild LR scheduler with the new optimizer
        if self.learning_rate_decay:
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=args.learning_rate_decay_gamma if hasattr(args, 'learning_rate_decay_gamma') else 0.99
            )

    def set_parameters(self, model):
        """Override to exclude BN layers (FedBN principle, same as clientBN)."""
        for (nn, np_param), (on, op_param) in zip(
            model.named_parameters(), self.model.named_parameters()
        ):
            if 'bn' not in nn:
                op_param.data = np_param.data.clone()

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        self.semantic_head.train()
        self.style_head.train()
        self.sem_classifier.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Online accumulators for prototypes (FIX: avoid storing all features on GPU)
        proto_sum = {}   # class -> sum of z_sem vectors
        proto_count = {} # class -> count
        style_sum = None
        style_sq_sum = None
        style_count = 0

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # Forward through backbone (base)
                h = self.model.base(x)  # h: [B, feat_dim]

                # Dual-head projection
                z_sem = self.semantic_head(h)   # [B, proj_dim]
                z_sty = self.style_head(h)      # [B, proj_dim]

                # === Loss 1: Task loss on SEMANTIC features (FIX: gradient flows through semantic_head) ===
                output = self.sem_classifier(z_sem)
                loss_task = self.loss(output, y)

                # === Loss 2: Task loss on augmented features ===
                loss_task_aug = torch.tensor(0.0, device=self.device)
                if self.style_bank is not None and self.current_round >= self.warmup_rounds:
                    h_aug = self._style_augment(h)
                    z_sem_aug = self.semantic_head(h_aug)
                    output_aug = self.sem_classifier(z_sem_aug)
                    loss_task_aug = self.loss(output_aug, y)

                # === Loss 3: Decoupling loss (FIX: independent weights) ===
                loss_orth, loss_hsic = self._compute_decouple_loss(z_sem, z_sty)

                # === Loss 4: Semantic contrastive alignment ===
                loss_sem_con = torch.tensor(0.0, device=self.device)
                if self.global_semantic_protos is not None:
                    loss_sem_con = self._compute_sem_con_loss(z_sem, y)

                # Total loss (FIX: lambda_orth and lambda_hsic are separate)
                loss = loss_task + loss_task_aug + \
                       self.lambda_orth * loss_orth + \
                       self.lambda_hsic * loss_hsic + \
                       self.lambda_sem * loss_sem_con

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Online prototype accumulation (last epoch only, on CPU)
                if epoch == max_local_epochs - 1:
                    z_sem_detached = z_sem.detach()
                    h_detached = h.detach()
                    for i, yy in enumerate(y):
                        c = yy.item()
                        feat_cpu = z_sem_detached[i].cpu()
                        if c not in proto_sum:
                            proto_sum[c] = feat_cpu
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += feat_cpu
                            proto_count[c] += 1

                    # Online Welford-style stats for style (FIX: no large tensor storage)
                    batch_mu = h_detached.mean(dim=0).cpu()
                    batch_sq = (h_detached ** 2).mean(dim=0).cpu()
                    b = h_detached.size(0)
                    if style_sum is None:
                        style_sum = batch_mu * b
                        style_sq_sum = batch_sq * b
                        style_count = b
                    else:
                        style_sum += batch_mu * b
                        style_sq_sum += batch_sq * b
                        style_count += b

        # Compute local semantic prototypes (per-class, FIX: count-aware)
        self.semantic_protos = {}
        self.semantic_proto_counts = {}
        for c in proto_sum:
            self.semantic_protos[c] = proto_sum[c] / proto_count[c]
            self.semantic_proto_counts[c] = proto_count[c]

        # Compute style prototype (per-domain)
        if style_count > 1:
            mu = style_sum / style_count
            var = style_sq_sum / style_count - mu ** 2
            self.style_proto_mu = mu
            self.style_proto_sigma = var.clamp(min=1e-6).sqrt()
        else:
            self.style_proto_mu = None
            self.style_proto_sigma = None

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _style_augment(self, h):
        """AdaIN-style augmentation using external style from bank."""
        idx = np.random.randint(0, len(self.style_bank))
        mu_ext, sigma_ext = self.style_bank[idx]
        mu_ext = mu_ext.to(self.device)
        sigma_ext = sigma_ext.to(self.device)

        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

        alpha = np.random.beta(0.1, 0.1)
        mu_mixed = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mixed = alpha * sigma_local + (1 - alpha) * sigma_ext

        h_norm = (h - mu_local) / sigma_local
        h_aug = h_norm * sigma_mixed + mu_mixed
        return h_aug

    def _compute_decouple_loss(self, z_sem, z_sty):
        """Orthogonal + HSIC dual constraint. Returns (loss_orth, loss_hsic) separately."""
        z_sem_norm = F.normalize(z_sem, dim=1)
        z_sty_norm = F.normalize(z_sty, dim=1)
        cos_sim = (z_sem_norm * z_sty_norm).sum(dim=1)
        loss_orth = (cos_sim ** 2).mean()

        loss_hsic = self._hsic(z_sem, z_sty)
        return loss_orth, loss_hsic

    def _hsic(self, x, y):
        """HSIC with Gaussian kernel and numerical guards."""
        n = x.size(0)
        if n < 4:
            return torch.tensor(0.0, device=self.device)

        Kx = self._gaussian_kernel(x)
        Ky = self._gaussian_kernel(y)

        H = torch.eye(n, device=self.device) - torch.ones(n, n, device=self.device) / n
        hsic = torch.trace(Kx @ H @ Ky @ H) / (n * n)
        return hsic

    def _gaussian_kernel(self, x):
        """Gaussian kernel with median heuristic and numerical guard."""
        n = x.size(0)
        dist = torch.cdist(x, x, p=2) ** 2

        # FIX: Guard against all-zero distances
        nonzero_dist = dist[dist > 0]
        if nonzero_dist.numel() == 0:
            return torch.ones(n, n, device=self.device)
        median_dist = torch.median(nonzero_dist)
        bandwidth = median_dist / (2.0 * np.log(n + 1) + 1e-6)
        K = torch.exp(-dist / (2.0 * bandwidth.clamp(min=1e-6)))
        return K

    def _compute_sem_con_loss(self, z_sem, y):
        """Vectorized InfoNCE: pull toward same-class global proto, push away others."""
        # Build prototype matrix [C_avail, proj_dim]
        available_classes = sorted([c for c, p in self.global_semantic_protos.items() if p is not None])
        if len(available_classes) < 2:
            return torch.tensor(0.0, device=self.device)

        proto_matrix = torch.stack([
            self.global_semantic_protos[c].to(self.device) for c in available_classes
        ])  # [C_avail, proj_dim]
        class_to_idx = {c: i for i, c in enumerate(available_classes)}

        # Cosine similarity: [B, C_avail]
        z_norm = F.normalize(z_sem, dim=1)
        p_norm = F.normalize(proto_matrix, dim=1)
        logits = (z_norm @ p_norm.T) / self.tau  # [B, C_avail]

        # Build target indices
        targets = []
        valid_mask = []
        for i in range(y.size(0)):
            label = y[i].item()
            if label in class_to_idx:
                targets.append(class_to_idx[label])
                valid_mask.append(i)

        if len(valid_mask) == 0:
            return torch.tensor(0.0, device=self.device)

        valid_mask = torch.tensor(valid_mask, device=self.device)
        targets = torch.tensor(targets, device=self.device, dtype=torch.long)
        logits_valid = logits[valid_mask]

        loss = F.cross_entropy(logits_valid, targets)
        return loss

    def set_style_bank(self, style_bank):
        self.style_bank = style_bank

    def set_global_semantic_protos(self, protos):
        self.global_semantic_protos = protos

    def set_round(self, round_num):
        self.current_round = round_num

    def get_semantic_head_params(self):
        """Return semantic head + classifier parameters for aggregation."""
        params = {}
        for k, v in self.semantic_head.state_dict().items():
            params['semantic_head.' + k] = copy.deepcopy(v)
        for k, v in self.sem_classifier.state_dict().items():
            params['sem_classifier.' + k] = copy.deepcopy(v)
        return params

    def set_semantic_head_params(self, params):
        """Load aggregated semantic head + classifier parameters."""
        head_params = {k.replace('semantic_head.', ''): v for k, v in params.items() if k.startswith('semantic_head.')}
        cls_params = {k.replace('sem_classifier.', ''): v for k, v in params.items() if k.startswith('sem_classifier.')}
        self.semantic_head.load_state_dict(head_params)
        self.sem_classifier.load_state_dict(cls_params)

    def test_metrics(self):
        """FIX: Evaluate using semantic_head + sem_classifier path (same as training)."""
        testloaderfull = self.load_test_data()
        self.model.eval()
        self.semantic_head.eval()
        self.sem_classifier.eval()

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                h = self.model.base(x)
                z_sem = self.semantic_head(h)
                output = self.sem_classifier(z_sem)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num, 0
