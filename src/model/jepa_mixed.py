# jepa_mixed.py
from copy import deepcopy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from src.model.position_hints import PositionHintComposer, PositionHintConfig
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, expansion=2, dropout=0.1):
        super().__init__()
        inner = int(expansion * d_model)
        # GEGLU: xW_g ⊙ σ(xW_gate)
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, inner * 2)  # split -> [inner, inner]
        self.act = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner, d_model)

        # init fan-in
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z = self.ln(x)
        a, g = self.fc1(z).chunk(2, dim=-1)  # [N,inner], [N,inner]
        z = a * self.act(g)  # GEGLU
        z = self.drop(z)
        z = self.fc2(z)
        return x + z


class PredictorMLP(nn.Module):
    def __init__(
        self, d_model: int, depth: int = 2, expansion: float = 2.0, dropout: float = 0.1
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model, expansion, dropout) for _ in range(depth)]
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):  # [N, D]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_out(x)


class PredictorMicroAttn(nn.Module):
    def __init__(
        self, d_model: int, nhead: int = 4, window: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.window = window
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.mlp = PredictorMLP(d_model, depth=2, expansion=2.0, dropout=dropout)

    def forward(
        self, h_ctx_full, idx_masked, hint
    ):  # h_ctx_full:[L,D], idx:[M], hint:[M,D]
        # Lấy cửa sổ lân cận cho mỗi idx (pad biên bằng chính token biên)
        L, D = h_ctx_full.shape
        neighborhoods = []
        for i in idx_masked.tolist():
            left = max(0, i - self.window)
            right = min(L, i + self.window + 1)
            neigh = h_ctx_full[left:right]  # [T,D], T<=2w+1
            neighborhoods.append(neigh)
        # batchify bằng pad
        T_max = max(n.shape[0] for n in neighborhoods)
        kv = h_ctx_full.new_zeros((len(neighborhoods), T_max, D))
        mask = torch.ones(
            (len(neighborhoods), T_max), dtype=torch.bool, device=h_ctx_full.device
        )
        for b, n in enumerate(neighborhoods):
            t = n.shape[0]
            kv[b, :t] = n
            mask[b, :t] = False  # False = keep
        q = self.ln_q(hint).unsqueeze(1)  # [M,1,D]
        kv = self.ln_kv(kv)  # [M,T,D]
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=mask)  # [M,1,D]
        x = attn_out.squeeze(1)  # [M,D]
        return self.mlp(x)  # [M,D]


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float = 0.996):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))


class MixedNLFOLJEPA(nn.Module):
    """
    JEPA cho NL↔FOL:
      - online: SATEWithFusion (masked)
      - target: SATEWithFusion (full, stopgrad, EMA)
      - predictor: dự đoán embedding từng token bị mask
      - aux losses: SA, DS, GR, XL
    """

    def __init__(
        self,
        online_encoder,
        target_encoder,
        d_model: int,
        loss_type: str = "mse",
        use_bias_stats: bool = True,
        # hệ số loss phụ:
        lambda_sa: float = 0.05,  # Structural–Attention agreement
        lambda_ds: float = 0.10,  # Dual-Stream consistency
        lambda_gr: float = 0.01,  # Gating regularization
        lambda_xl: float = 0.10,  # Cross-Modal lexical alignment
    ):
        super().__init__()
        self.online = online_encoder
        self.target = target_encoder
        self.target.load_state_dict(self.online.state_dict(), strict=True)
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = PredictorMicroAttn(d_model=d_model)
        self.loss_type = loss_type

        bias_ch = getattr(getattr(self.online, "struct", None), "cfg", None)
        bias_ch = getattr(bias_ch, "bias_channels", 0) if bias_ch else 0
        self.hint_comp = PositionHintComposer(
            sate_encoder=self.online.sate,
            d_model=d_model,
            cfg=PositionHintConfig(use_bias_stats=use_bias_stats, dropout=0.1),
            bias_channels=bias_ch,
        )
        self.hint_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.hint_proj.weight)
        self.hint_alpha = nn.Parameter(torch.tensor(1.0))

        self.bias_builder = getattr(self.online, "bias_builder", None)

        # lambdas
        self.lambda_sa = lambda_sa
        self.lambda_ds = lambda_ds
        self.lambda_gr = lambda_gr
        self.lambda_xl = lambda_xl

    def _cosine_loss(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return 1.0 - (a * b).sum(dim=-1).mean()

    def forward(self, item_full: Dict, mask_flags: List[int], ema_m: float = 0.996):
        device = next(self.parameters()).device

        # 1) masked item & encode
        item_ctx = dict(item_full)
        item_ctx["mask_flags"] = mask_flags
        out_ctx = self.online(
            item_ctx
        )  # {'h_fused','h_lin','h_path','semantic_mask','meta'}
        with torch.no_grad():
            out_tgt = self.target(item_full)  # full

        h_ctx, h_tgt = out_ctx["h_fused"], out_tgt["h_fused"].detach()  # [L,D]
        L = h_ctx.size(0)

        # 2) idx masked & meaningful
        mask = torch.tensor(mask_flags, device=device, dtype=torch.bool)
        if "semantic_mask" in out_ctx:
            mask &= out_ctx["semantic_mask"].to(device).bool()
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            with torch.no_grad():
                ema_update(self.target, self.online, m=ema_m)
            return {
                "loss": h_ctx.new_zeros(()),
                "num_masked": torch.tensor(0, device=device),
            }

        # 3) JEPA main prediction với position-hint
        raw_bias = None
        if callable(self.bias_builder):
            B_init, _ = self.bias_builder(item_full)  # [L,L,C]
            raw_bias = B_init.to(h_ctx.device)
        hint_LD = self.hint_comp(out_tgt["meta"], raw_bias)  # [L,D]

        x = h_ctx.index_select(0, idx)  # [M,D]
        hint_masked = hint_LD.index_select(0, idx)  # [M,D]
        # x_plus = x + self.hint_alpha * self.hint_proj(hint)

        z_pred = self.predictor(h_ctx_full=h_ctx, idx_masked=idx, hint=hint_masked)
        z_tgt = h_tgt.index_select(0, idx)

        if self.loss_type == "mse":
            loss_main = F.mse_loss(z_pred, z_tgt)
        else:
            loss_main = self._cosine_loss(z_pred, z_tgt)

        loss = loss_main
        logs = {"loss_main": loss_main.detach(), "num_masked": idx.numel()}

        # ================== Aux A: Dual-Stream consistency (DS) ==================
        if self.lambda_ds > 0:
            # symmetrized MSE giữa 2 stream (detach mỗi bên một lần)
            ds = F.mse_loss(out_ctx["h_lin"], out_ctx["h_path"].detach()) + F.mse_loss(
                out_ctx["h_path"], out_ctx["h_lin"].detach()
            )
            loss = loss + self.lambda_ds * ds
            logs["loss_ds"] = ds.detach()

        # ================== Aux B: Gating regularization (GR) ==================
        if self.lambda_gr > 0 and hasattr(self.online.fusion, "alpha"):
            alpha = self.online.fusion.alpha
            gate = torch.sigmoid(alpha)  # per-dim hoặc scalar
            p0 = 0.12  # gần σ(-2) → ưu tiên h_lin đầu train
            gr = F.mse_loss(gate, gate.new_full(gate.shape, p0))
            loss = loss + self.lambda_gr * gr
            logs["loss_gr"] = gr.detach()

        # ================== Aux C: Cross-Modal lexical alignment (XL) ===========
        if self.lambda_xl > 0:
            seg_meta = item_full["seg_meta"]  # [(seg_id, modality)]
            toks = [t for t, _ in item_full["tokens"]]
            mod = torch.tensor([m for _, m in seg_meta], device=device)  # 0 NL, 1 FOL

            # chỉ tạo cặp trong sample hiện tại
            pairs = []
            table = {}
            for i, t in enumerate(toks):
                if t in {"(", ")"}:
                    continue
                key = t.lower()
                table.setdefault((key, int(mod[i].item())), []).append(i)
            for (key, m0), idxs0 in table.items():
                m1 = 1 - m0
                if (key, m1) in table:
                    for i in idxs0:
                        for j in table[(key, m1)]:
                            pairs.append((i, j))

            xl = h_ctx.new_zeros(())
            if pairs:
                i_idx = torch.tensor([i for i, _ in pairs], device=device)
                j_idx = torch.tensor([j for _, j in pairs], device=device)
                # cosine loss giữa cặp NL↔FOL trùng bề mặt
                xl = self._cosine_loss(
                    h_ctx.index_select(0, i_idx), h_ctx.index_select(0, j_idx)
                )
                loss = loss + self.lambda_xl * xl
            logs["loss_xl"] = xl.detach()

        # ================== Aux D: Structural–Attention agreement (SA) =========
        if self.lambda_sa > 0 and raw_bias is not None:
            # mục tiêu Q_ij = softmax(scores_bias[i,j]) theo j
            with torch.no_grad():
                # scores từ raw_bias và tham số (γ, w)
                scores = torch.tensordot(
                    raw_bias, self.online.struct.w, dims=([2], [0])
                )  # [L,L]
                scores = self.online.struct.gamma * scores
                Q = F.softmax(scores, dim=-1).clamp_min(1e-9)  # [L,L]

            # lấy attention P từ online.struct.last_attn (list per layer, mỗi cái [B=1,L,L])
            P_list = getattr(self.online.struct, "last_attn", None)
            if P_list:
                # vì batch=1 → squeeze(0)
                kl = 0.0
                for P in P_list:
                    P = P[0].clamp_min(1e-9)  # [L,L]
                    kl += (
                        (P * (P.log() - Q.log())).sum(dim=-1).mean()
                    )  # KL(P||Q) theo i
                kl = kl / len(P_list)
                loss = loss + self.lambda_sa * kl
                logs["loss_sa"] = kl.detach()

        # EMA update target
        with torch.no_grad():
            ema_update(self.target, self.online, m=ema_m)

        return {"loss": loss, **logs}
