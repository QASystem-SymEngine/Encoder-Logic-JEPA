# position_hints.py
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass
class PositionHintConfig:
    use_bias_stats: bool = True  # dùng thống kê từ raw_bias [L,L,C]
    dropout: float = 0.1


class PositionHintComposer(nn.Module):
    """
    Sinh 'position-hint' p ∈ R^{L×D} từ meta của encoder (pos/depth/seg/mod/node-type)
    và (tuỳ chọn) thống kê bias theo kênh từ raw_bias [L,L,C].
    """

    def __init__(
        self,
        sate_encoder,
        d_model: int,
        cfg: PositionHintConfig,
        bias_channels: int = 0,
    ):
        super().__init__()
        self.sate = sate_encoder  # để dùng lại các embedding: pos_emb, depth_emb, seg_emb, modality_emb, node_type_emb
        self.cfg = cfg

        self.bias_proj = (
            nn.Linear(bias_channels, d_model)
            if (cfg.use_bias_stats and bias_channels > 0)
            else None
        )
        # Trộn & chuẩn hoá nhẹ để ổn định
        self.mix = nn.Linear(d_model, d_model, bias=False)  # map D->D, init identity
        nn.init.eye_(self.mix.weight)
        self.ln = nn.LayerNorm(d_model)
        self.do = nn.Dropout(cfg.dropout)
        # Gate cho toàn hint (có thể học âm/dương)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, meta: Dict[str, torch.Tensor], raw_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        meta:  dict từ SATEWithFusion(...)"meta" của target (full, không mask)
               cần có: positions[L], depths[L], seg_ids[L], modality_ids[L], node_type_ids[L]
        raw_bias: [L, L, C] nếu muốn dùng bias stats
        return:  hint [L, D]
        """
        pos_ids = meta["positions"]
        depths = meta["depths"]
        seg_ids = meta.get("seg_ids")
        mod_ids = meta.get("modality_ids")
        ntype_ids = meta.get("node_type_ids")

        # Các embedding định vị/cấu trúc (D chiều mỗi cái)
        e_pos = self.sate.pos_emb(pos_ids)
        e_depth = self.sate.depth_emb(depths)
        e_seg = self.sate.seg_emb(seg_ids) if seg_ids is not None else 0
        e_mod = self.sate.modality_emb(mod_ids) if mod_ids is not None else 0
        e_ntype = self.sate.node_type_emb(ntype_ids) if ntype_ids is not None else 0

        # Tổng hợp (không dùng lex/type/value T5 để tránh rò rỉ nội dung)
        h = e_pos + e_depth + e_seg + e_mod + e_ntype  # [L, D]

        # (Tuỳ chọn) thêm thống kê bias theo kênh
        if self.bias_proj is not None and raw_bias is not None:
            # raw_bias: [L, L, C] → lấy mean theo trục j
            stats = raw_bias.mean(dim=1)  # [L, C]
            h = h + self.bias_proj(stats)  # [L, D]

        # Chuẩn hoá & mix tuyến tính nhẹ
        h = self.ln(self.do(h))
        h = self.mix(h) * self.gate
        return h  # [L, D]
