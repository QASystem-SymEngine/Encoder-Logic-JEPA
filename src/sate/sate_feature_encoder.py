# sate_feature_encoder.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast


@dataclass
class SATEConfig:
    t5_name: str = "t5-base"
    d_ast: int = 768  # = d_model của t5-base
    max_seq_len: int = 512
    max_depth: int = 128
    dropout: float = 0.1
    max_segments: int = 10  # len(ast_nl) + len(ast_fol)
    node_type_vocab: Tuple[str, ...] = (
        "FORALL",
        "EXISTS",
        "IFF",
        "IMPLIES",
        "OR",
        "AND",
        "XOR",
        "NOT",
        "Variable",
        "Predicate",
        "NLToken",
    )


# --------------------------- Helper ---------------------------
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)  # [B, L, 1]
    x = x * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return x.sum(dim=1) / denom  # [B, H]


class SATEFeatureEncoder(nn.Module):

    def __init__(
        self,
        cfg: SATEConfig,
        tokenizer: T5TokenizerFast,
        t5_model: T5ForConditionalGeneration,
    ):
        """
        Args:
            cfg: cấu hình
            tokenizer: T5TokenizerFast (truyền từ bên ngoài)
            t5_model: T5ForConditionalGeneration (truyền từ bên ngoài, dùng encoder)
        """
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.t5 = t5_model.encoder

        d_t5 = self.t5.config.d_model
        assert d_t5 == cfg.d_ast, f"d_ast ({cfg.d_ast}) phải bằng d_t5 ({d_t5})."

        # --- Linear mỏng cùng chiều (khởi tạo Identity) ---
        self.proj_lex = nn.Linear(d_t5, cfg.d_ast, bias=False)
        self.proj_type = nn.Linear(d_t5, cfg.d_ast, bias=False)
        self.proj_value = nn.Linear(d_t5, cfg.d_ast, bias=False)
        nn.init.eye_(self.proj_lex.weight)  # bắt đầu như passthrough
        nn.init.eye_(self.proj_type.weight)
        nn.init.eye_(self.proj_value.weight)

        # --- Embeddings bổ sung ---
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_ast)
        self.depth_emb = nn.Embedding(cfg.max_depth + 1, cfg.d_ast)

        self.ntype2id: Dict[str, int] = {
            n: i for i, n in enumerate(cfg.node_type_vocab)
        }
        self.node_type_emb = nn.Embedding(len(self.ntype2id), cfg.d_ast)

        self.dropout = nn.Dropout(cfg.dropout)
        self.ln = nn.LayerNorm(cfg.d_ast)

        self.seg_emb = nn.Embedding(cfg.max_segments, cfg.d_ast)
        self.modality_emb = nn.Embedding(2, cfg.d_ast)  # 0=NL, 1=FOL

    # ---- T5 sentence embedding (mean pooling) ----
    def _t5_embed(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        batch = {k: v.to(self.t5.device) for k, v in batch.items()}
        with torch.set_grad_enabled(True):
            out = self.t5(**batch).last_hidden_state  # [B, L, d_t5]
        return _masked_mean(out, batch["attention_mask"])  # [B, d_t5]

    def forward(self, item: Dict) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        tokens: List[Tuple[str, int]] = item["tokens"]
        L = len(tokens)
        seg_meta: List[Tuple[int, int]] = item.get("seg_meta", [(0, 1)] * L)

        type_paths = {p["current_id"]: p for p in item.get("type_paths", [])}
        value_paths = {p["current_id"]: p for p in item.get("value_paths", [])}

        # NEW: mask_flags (0/1) dài L; 1 => mask token i
        mask_flags = item.get("mask_flags", [0] * L)
        assert len(mask_flags) == L, "mask_flags length mismatch"
        mask_flags = torch.tensor(mask_flags, device=device, dtype=torch.long)

        pos_ids = torch.arange(L, device=device).clamp(max=self.cfg.max_seq_len - 1)
        seg_ids = torch.tensor([s for s, _ in seg_meta], device=device).clamp(
            max=self.cfg.max_segments - 1
        )
        mod_ids = torch.tensor([m for _, m in seg_meta], device=device)

        semantic_mask = torch.tensor(
            [0 if tok in {"(", ")"} else 1 for tok, _ in tokens],
            device=device,
            dtype=torch.long,
        )

        depths, node_type_ids = [], []
        lex_texts, type_texts, value_texts = [], [], []

        for i, (tok, gid) in enumerate(tokens):
            # bracket luôn vô nghĩa
            if tok in {"(", ")"}:
                depths.append(0)
                node_type_ids.append(self.ntype2id["NLToken"])
                lex_texts.append("")
                type_texts.append("")
                value_texts.append("")
                continue

            # Nếu token bị mask → triệt tiêu lex/type/value
            if mask_flags[i] == 1:
                depths.append(0)  # đơn giản hoá: depth=0 cho masked
                node_type_ids.append(self.ntype2id["NLToken"])
                lex_texts.append("")
                type_texts.append("")
                value_texts.append("")
                continue

            t_entry = type_paths.get(gid)
            v_entry = value_paths.get(gid)

            d = (
                max(
                    len(t_entry["paths"]) if (t_entry and t_entry.get("paths")) else 1,
                    len(v_entry["paths"]) if (v_entry and v_entry.get("paths")) else 1,
                )
                - 1
            )
            depths.append(min(d, self.cfg.max_depth))

            ntype = None
            if t_entry and t_entry.get("paths"):
                cand = t_entry["paths"][-1]
                if cand in self.ntype2id:
                    ntype = cand
            node_type_ids.append(self.ntype2id.get(ntype, self.ntype2id["NLToken"]))

            lex_texts.append(tok)
            type_texts.append(
                " ".join(t_entry["paths"])
                if (t_entry and t_entry.get("paths"))
                else tok
            )
            value_texts.append(
                " ".join(v_entry["paths"])
                if (v_entry and v_entry.get("paths"))
                else tok
            )

        depths = torch.tensor(depths, device=device, dtype=torch.long)
        node_type_ids = torch.tensor(node_type_ids, device=device, dtype=torch.long)

        e_lex_t5 = self._t5_embed(lex_texts)
        e_type_t5 = self._t5_embed(type_texts)
        e_value_t5 = self._t5_embed(value_texts)

        e_lex = self.proj_lex(e_lex_t5)
        e_type = self.proj_type(e_type_t5)
        e_value = self.proj_value(e_value_t5)

        e_pos = self.pos_emb(pos_ids)
        e_depth = self.depth_emb(depths)
        e_ntype = self.node_type_emb(node_type_ids)
        e_seg = self.seg_emb(seg_ids)
        e_modal = self.modality_emb(mod_ids)

        sem = semantic_mask.unsqueeze(-1).float()
        e_lex, e_type, e_value, e_depth, e_ntype = (
            e_lex * sem,
            e_type * sem,
            e_value * sem,
            e_depth * sem,
            e_ntype * sem,
        )

        h0 = e_lex + e_type + e_value + e_depth + e_ntype + e_pos + e_seg + e_modal
        h0 = self.ln(self.dropout(h0))

        return {
            "h0": h0,
            "semantic_mask": semantic_mask,
            "meta": {
                "positions": pos_ids,
                "depths": depths,
                "node_type_ids": node_type_ids,
                "seg_ids": seg_ids,
                "modality_ids": mod_ids,
                "mask_flags": mask_flags,
            },
        }
