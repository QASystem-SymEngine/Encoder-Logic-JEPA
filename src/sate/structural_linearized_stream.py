# structural_linearized_stream.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from typing import Set

# ===================== Bias builder =====================


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    k = 0
    for i in range(n):
        if a[i] == b[i]:
            k += 1
        else:
            break
    return k


def build_structure_bias_from_paths(
    item: Dict,
    prefer: str = "value_paths",  # "value_paths" | "type_paths"
    max_clip: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo structure bias B ∈ R^{L×L} từ path_ids:
      depth_i   = len(path_ids_i) - 1
      lca_ij    = len(common_prefix(path_i, path_j)) - 1
      abs_diff  = |depth_i - depth_j|
      bias_ij   = w_lca * lca_ij - w_dd * abs_diff  (các w sẽ là tham số learnable của model)

    Trả về:
      - B_init:  [L, L] (float tensor, KHÔNG có w; chỉ là raw features ghép 2 kênh)
      - depths:  [L] (long tensor)
    """
    tokens: List[Tuple[str, int]] = item["tokens"]
    L = len(tokens)

    # chọn nguồn path
    paths_list = item.get(prefer, []) or item.get("type_paths", [])
    mp = {p["current_id"]: p["path_ids"] for p in paths_list}

    # chuẩn hóa path cho từng token (ngoặc -> [])
    paths_per_tok: List[List[int]] = []
    depths = []
    for tok, idx in tokens:
        path_ids = mp.get(idx, [])
        d = max(len(path_ids) - 1, 0)
        paths_per_tok.append(path_ids)
        depths.append(d)

    depths_t = torch.tensor(depths, dtype=torch.long)

    # tính raw features: lca_depth & abs_depth_diff
    LCA = torch.zeros((L, L), dtype=torch.float)
    ABS = torch.zeros((L, L), dtype=torch.float)
    for i in range(L):
        for j in range(L):
            lca = _common_prefix_len(paths_per_tok[i], paths_per_tok[j]) - 1
            lca = max(lca, 0)
            LCA[i, j] = float(lca)
            ABS[i, j] = float(abs(depths[i] - depths[j]))

    # Chuẩn hóa nhẹ để ổn định (clip)
    LCA = LCA.clamp_(0, max_clip)
    ABS = ABS.clamp_(0, max_clip)

    # Ghép 2 kênh thành 1 tensor để model tự học hệ số (w_lca, w_dd)
    # (ta sẽ tuyến tính hoá trong StructuralTransformer.forward)
    B_init = torch.stack([LCA, ABS], dim=-1)  # [L, L, 2]
    return B_init, depths_t


def build_mixed_bias(
    item: Dict, max_clip: float = 6.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    raw_bias: [L, L, C] với C=4 kênh: [LCA, ABS, ADJ, LEX]
    depths: [L]
    """
    tokens: List[Tuple[str, int]] = item["tokens"]
    seg_meta: List[Tuple[int, int]] = item["seg_meta"]
    L = len(tokens)

    # ---- LCA & ABS (tái dùng code cũ, nhưng đọc path_ids từ type/value_paths bất kỳ) ----
    # Ghép map path_ids theo ưu tiên value_paths > type_paths (để NL vẫn có)
    paths_list = item.get("value_paths", []) or item.get("type_paths", [])
    mp = {p["current_id"]: (p.get("path_ids") or []) for p in paths_list}

    paths_per_tok: List[List[int]] = []
    depths = []
    for tok, gid in tokens:
        path_ids = mp.get(gid, [])
        d = max(len(path_ids) - 1, 0)
        paths_per_tok.append(path_ids)
        depths.append(d)

    depths_t = torch.tensor(depths, dtype=torch.long)
    LCA = torch.zeros((L, L), dtype=torch.float)
    ABS = torch.zeros((L, L), dtype=torch.float)
    for i in range(L):
        for j in range(L):
            # common prefix len
            a, b = paths_per_tok[i], paths_per_tok[j]
            n = min(len(a), len(b))
            k = 0
            for t in range(n):
                if a[t] == b[t]:
                    k += 1
                else:
                    break
            lca = max(k - 1, 0)
            LCA[i, j] = float(lca)
            ABS[i, j] = float(abs(depths[i] - depths[j]))

    LCA.clamp_(0, max_clip)
    ABS.clamp_(0, max_clip)

    # ---- ADJ (segment adjacency prior) ----
    seg_ids = [s for s, _ in seg_meta]
    ADJ = torch.zeros((L, L), dtype=torch.float)
    for i in range(L):
        si = seg_ids[i]
        for j in range(L):
            sj = seg_ids[j]
            if abs(si - sj) <= 1:  # cùng seg hoặc láng giềng
                ADJ[i, j] = 1.0

    # ---- LEX (lexical surface tie NL<->FOL) ----
    # chỉ boost khi modality khác nhau và bề mặt trùng
    mod_ids = [m for _, m in seg_meta]
    toks_norm = [t[0].lower() for t in tokens]
    LEX = torch.zeros((L, L), dtype=torch.float)
    for i in range(L):
        for j in range(L):
            if (
                mod_ids[i] != mod_ids[j]
                and toks_norm[i] == toks_norm[j]
                and toks_norm[i] not in {"(", ")"}
            ):
                LEX[i, j] = 1.0

    B_init = torch.stack([LCA, ABS, ADJ, LEX], dim=-1)  # [L, L, 4]
    return B_init, depths_t


# ===================== Structural Transformer =====================


@dataclass
class StructuralTransformerConfig:
    d_model: int = 768
    nhead: int = 8
    num_layers: int = 2
    dim_ff: int = 2048
    dropout: float = 0.1
    bias_scale_init: float = 1.0
    bias_channels: int = 4  # 2 (LCA, ABS) hoặc 4 (LCA, ABS, ADJ, LEX)


class StructuralTransformerLayer(nn.Module):
    def __init__(self, cfg: StructuralTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, attn_bias: torch.Tensor, need_weights: bool = False
    ):
        """
        x: [B, L, D]
        attn_bias: [B, L, L]  (additive bias vào scores)
        return: (x_out, attn_probs or None)
        """
        B, L, _ = x.shape
        # Expand attn_bias -> [B*nH, L, L] theo quy ước attn_mask float của PyTorch
        bias = attn_bias.to(dtype=x.dtype)
        bias = (
            bias.unsqueeze(1)
            .expand(B, self.cfg.nhead, L, L)
            .reshape(B * self.cfg.nhead, L, L)
        )
        attn_out, attn_w = self.mha(
            x,
            x,
            x,
            attn_mask=bias,
            need_weights=need_weights,
            average_attn_weights=False,  # lấy theo từng head
        )
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        if need_weights:
            # attn_w shape: [B, nH, L, L]
            # MultiheadAttention trả về [nH*B, L, L] khi không batch_first; với batch_first=True + average_attn_weights=False → [B, nH, L, L]
            if attn_w.dim() == 3:
                # fallback PyTorch cũ: [B*nH, L, L] -> reshape
                attn_w = attn_w.view(B, self.cfg.nhead, L, L)
            return x, attn_w
        return x, None


class StructuralTransformer(nn.Module):
    """
    score_ij := score_ij + gamma * sum_c( w[c] * raw_bias_ij[c] )
    """

    def __init__(self, cfg: StructuralTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [StructuralTransformerLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.w = nn.Parameter(torch.zeros(cfg.bias_channels))
        with torch.no_grad():
            if cfg.bias_channels >= 1:
                self.w[0] = 1.0  # LCA +
            if cfg.bias_channels >= 2:
                self.w[1] = -1.0  # -ABS
        self.gamma = nn.Parameter(torch.tensor(cfg.bias_scale_init))

        # nơi lưu attention cuối forward (dùng cho SA loss)
        self.last_attn = []  # List[Tensor [B, L, L]] per layer (avg head)

    def _make_attn_bias(self, raw_bias: torch.Tensor) -> torch.Tensor:
        # raw_bias [B,L,L,C] · w[C] -> [B,L,L]
        return self.gamma * torch.tensordot(raw_bias, self.w, dims=([3], [0]))

    def forward(self, x: torch.Tensor, raw_bias: torch.Tensor) -> torch.Tensor:
        """
        x:        [B, L, D]
        raw_bias: [B, L, L, C]
        """
        self.last_attn = []
        attn_bias = self._make_attn_bias(raw_bias)  # [B, L, L]
        for layer in self.layers:
            x, attn_w = layer(x, attn_bias, need_weights=True)
            if attn_w is not None:
                # trung bình theo head → [B, L, L]
                self.last_attn.append(attn_w.mean(dim=1))
        return x
