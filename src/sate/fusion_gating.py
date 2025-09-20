# fusion_gating.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from sate.structural_linearized_stream import build_structure_bias_from_paths

# -------------------- Config --------------------


@dataclass
class FusionConfig:
    per_dim_alpha: bool = True  # True: alpha ∈ R^D; False: alpha là scalar
    dropout: float = 0.1


# -------------------- Helper --------------------


def _is_bracket(tok: str) -> bool:
    return tok in {"(", ")"}


def _build_type_value_texts(item: Dict) -> Tuple[List[str], List[str], List[int]]:
    """
    Tạo list text cho type_paths và value_paths (mỗi token một câu),
    đồng thời trả về semantic_mask (1: meaningful; 0: bracket).
    """
    tokens: List[Tuple[str, int]] = item["tokens"]
    type_map = {p["current_id"]: p for p in item.get("type_paths", [])}
    value_map = {p["current_id"]: p for p in item.get("value_paths", [])}

    type_texts, value_texts, sem_mask = [], [], []
    for tok, idx in tokens:
        if _is_bracket(tok):
            type_texts.append("")  # để pooling ra ~0
            value_texts.append("")
            sem_mask.append(0)
            continue
        t_entry = type_map.get(idx)
        v_entry = value_map.get(idx)
        type_texts.append(" ".join(t_entry["paths"]) if t_entry else tok)
        value_texts.append(" ".join(v_entry["paths"]) if v_entry else tok)
        sem_mask.append(1)
    return type_texts, value_texts, sem_mask


# -------------------- Fusion (gating) --------------------


class FusionGating(nn.Module):
    """
    Kết hợp Linearized stream và Path stream:
        h = LN(h_lin + sigmoid(alpha) ⊙ h_path)
    - h_lin: [B, L, D] từ StructuralTransformer
    - h_path: [B, L, D] tính từ type/value paths (dùng cùng T5 + projector của SATEFeatureEncoder)
    """

    def __init__(self, d_model: int, cfg: FusionConfig = FusionConfig()):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg.dropout)
        # alpha: per-dim hay scalar
        if cfg.per_dim_alpha:
            self.alpha = nn.Parameter(torch.zeros(d_model))  # khởi tạo 0 -> σ(0)=0.5
        else:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h_lin: torch.Tensor, h_path: torch.Tensor) -> torch.Tensor:
        """
        h_lin, h_path: [B, L, D]
        """
        # broadcast alpha
        if self.alpha.dim() == 0:
            gate = torch.sigmoid(self.alpha).view(1, 1, 1)
        else:
            gate = torch.sigmoid(self.alpha).view(1, 1, -1)
        h = h_lin + gate * h_path
        return self.ln(self.dropout(h))


# -------------------- Wrapper: kết nối 3 khối --------------------

from typing import Callable


class SATEWithFusion(nn.Module):
    def __init__(
        self,
        sate_encoder,
        structural_transformer,
        fusion_cfg: FusionConfig = FusionConfig(),
        bias_builder: Callable[[Dict], Tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        super().__init__()
        self.sate = sate_encoder
        self.struct = structural_transformer
        self.fusion = FusionGating(d_model=self.sate.cfg.d_ast, cfg=fusion_cfg)
        self.bias_builder = bias_builder

    def _path_stream_from_item(self, item: Dict) -> torch.Tensor:
        # (đã bỏ @no_grad theo patch trước)
        type_texts, value_texts, sem_mask = _build_type_value_texts(item)
        e_type_t5 = self.sate._t5_embed(type_texts)
        e_value_t5 = self.sate._t5_embed(value_texts)
        e_type = self.sate.proj_type(e_type_t5)
        e_value = self.sate.proj_value(e_value_t5)
        h_path = e_type + e_value
        sem = torch.tensor(
            sem_mask, device=h_path.device, dtype=torch.float32
        ).unsqueeze(-1)
        h_path = self.sate.ln(self.sate.dropout(h_path * sem))
        return h_path.unsqueeze(0)

    def forward(self, item: Dict) -> Dict[str, torch.Tensor]:
        out0 = self.sate(item)
        h0 = out0["h0"].unsqueeze(0)  # [1,L,D]

        if self.bias_builder is not None:
            B_init, _ = self.bias_builder(item)  # [L,L,C]
        else:
            B_init, _ = build_structure_bias_from_paths(item)

        h_lin = self.struct(h0, B_init.unsqueeze(0).to(h0.device))  # [1,L,D]
        h_path = self._path_stream_from_item(item)  # [1,L,D]
        h = self.fusion(h_lin, h_path)

        return {
            "h_fused": h.squeeze(0),
            "h_lin": h_lin.squeeze(0),
            "h_path": h_path.squeeze(0),
            "semantic_mask": out0["semantic_mask"],
            "meta": out0["meta"],
        }
