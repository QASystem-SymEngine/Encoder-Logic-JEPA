# context_fusion_head.py
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn


class ContextFusionHead(nn.Module):
    """
    Dùng offset_mapping để expand h_fused -> b khớp tuyệt đối với subword của context.
    - Tự dựng context_text từ tokens + seg_meta theo thứ tự NL⟂...⟂FOL.
    - Tokenize với return_offsets_mapping=True để lấy (char_start, char_end) của từng subword.
    - Với mỗi subword, tìm token i có span ký tự bao phủ → copy h_fused[i].
    - Áp dụng semantic_mask để triệt tiêu token ngoặc/bracket nếu có.
    """

    def __init__(
        self,
        d_model: int,
        use_proj: bool = True,
        dropout: float = 0.1,
        sep_token: str = " <extra_id_0> ",
        token_joiner: str = " ",
    ):
        super().__init__()
        self.proj = (
            nn.Linear(d_model, d_model, bias=False) if use_proj else nn.Identity()
        )
        if isinstance(self.proj, nn.Linear):
            nn.init.eye_(self.proj.weight)
        self.ln = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

        # Cấu hình dựng context
        self.sep_token = sep_token  # ngăn cách giữa các segment
        self.token_joiner = token_joiner  # nối các token trong 1 segment

    @staticmethod
    def _is_bracket(tok: str) -> bool:
        return tok in {"(", ")"}

    def _group_by_segment(
        self, tokens: List[Tuple[str, int]], seg_meta: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, str]]]:
        """
        Trả về list các segment; mỗi segment là list (global_index, token_str)
        """
        assert len(tokens) == len(seg_meta)
        groups: Dict[int, List[Tuple[int, str]]] = {}
        for i, ((tok, _), (seg_id, _mod)) in enumerate(zip(tokens, seg_meta)):
            groups.setdefault(seg_id, []).append((i, tok))
        return [groups[k] for k in sorted(groups.keys())]

    def _build_context_and_token_spans(
        self, tokens: List[Tuple[str, int]], seg_meta: List[Tuple[int, int]]
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Dựng context_text = seg0(token_joiner) + sep + seg1(...) + ...
        Đồng thời tạo spans_tok: với chiều dài = L (số token), mỗi phần tử là (char_start, char_end)
        của token trong context_text.
        """
        segments = self._group_by_segment(tokens, seg_meta)
        spans_tok: List[Tuple[int, int]] = [(-1, -1)] * len(tokens)

        parts: List[str] = []
        cursor = 0
        for si, seg in enumerate(segments):
            # nối token trong seg bằng token_joiner
            seg_text = self.token_joiner.join([tok for _, tok in seg])
            # ghi span cho từng token trong seg
            offset_in_seg = 0
            for gidx, tok in seg:
                start = cursor + offset_in_seg
                end = start + len(tok)
                spans_tok[gidx] = (start, end)
                offset_in_seg += len(tok)
                # nếu chưa phải token cuối cùng của seg thì cộng thêm 1 khoảng token_joiner
                if (gidx, tok) != seg[-1]:
                    offset_in_seg += len(self.token_joiner)
            parts.append(seg_text)
            # cập nhật cursor sau khi thêm seg + (nếu còn seg kế tiếp thì thêm sep)
            cursor += len(seg_text)
            if si != len(segments) - 1:
                parts.append(self.sep_token)
                cursor += len(self.sep_token)

        context_text = "".join(parts)
        return context_text, spans_tok

    def _expand_bias_by_offsets(
        self,
        tokenizer,
        context_text: str,
        h_fused: torch.Tensor,  # [L, D]
        token_spans: List[Tuple[int, int]],
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Tokenize context_text với return_offsets_mapping, rồi ánh xạ mỗi subword
        về token i bằng điều kiện bao phủ offset.
        Trả về:
          - b: [1, S, D]
          - extra: dict chứa input_ids, attention_mask để tái sử dụng ngoài head.
        """
        device = h_fused.device
        enc = tokenizer(
            context_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)  # [1, S]
        attn_mask = enc["attention_mask"].to(device)  # [1, S]
        offsets = enc["offset_mapping"][0].tolist()  # List[(s,e)] cho S vị trí

        L, D = h_fused.shape
        S = input_ids.shape[1]
        b = torch.zeros((S, D), device=device, dtype=h_fused.dtype)

        # Chuẩn bị mask lặp theo token -> subword
        if semantic_mask is not None:
            sem = semantic_mask.to(device).float()  # [L]
        else:
            sem = torch.ones(L, device=device)

        # Hàm tìm token index i cho 1 subword span (s,e)
        def locate_token(s: int, e: int) -> int:
            # Bỏ qua special tokens có (0,0) hoặc khoảng trắng
            if e <= s:
                return -1
            # Nhị phân tuyến tính đủ nhanh vì L thường << S
            for i, (ts, te) in enumerate(token_spans):
                # bao phủ hoặc giao cắt đủ lớn (ở đây dùng bao phủ chặt)
                if ts <= s and e <= te:
                    return i
            return -1

        for k, (s, e) in enumerate(offsets):
            idx = locate_token(s, e)
            if idx >= 0:
                b[k] = h_fused[idx] * sem[idx]
            else:
                # subword thuộc sep_token hoặc special → để 0
                pass

        b = self.proj(b).unsqueeze(0)  # [1, S, D]
        extras = {"input_ids": input_ids, "attention_mask": attn_mask}
        return b, extras

    def forward(
        self,
        # LƯU Ý: không cần context_hidden sẵn có nữa; head sẽ tự encode từ tokens+seg_meta
        h_fused: torch.Tensor,  # [L, D]
        tokens: List[Tuple[str, int]],
        seg_meta: List[Tuple[int, int]],
        tokenizer,
        semantic_mask: Optional[torch.Tensor] = None,
        t5_encoder=None,  # nếu truyền, head sẽ trả luôn context_hidden
    ) -> Dict[str, torch.Tensor]:
        """
        Trả về:
          - context_plus: [1, S, D]  (nếu có t5_encoder)
          - bias_expanded: [1, S, D]
          - context_hidden: [1, S, D] (nếu có t5_encoder)
          - input_ids, attention_mask: để dùng tiếp nếu muốn
          - context_text: chuỗi đã dựng (NL ⟂ ... ⟂ FOL)
        """
        # 1) Dựng context_text theo NL⟂...⟂FOL và tính span cho từng token
        context_text, token_spans = self._build_context_and_token_spans(
            tokens, seg_meta
        )

        # 2) Expand h_fused theo offsets của context_text
        b, extras = self._expand_bias_by_offsets(
            tokenizer=tokenizer,
            context_text=context_text,
            h_fused=h_fused,
            token_spans=token_spans,
            semantic_mask=semantic_mask,
        )  # b: [1,S,D]

        out = {
            "bias_expanded": b,  # [1, S, D]
            "input_ids": extras["input_ids"],  # [1, S]
            "attention_mask": extras["attention_mask"],  # [1, S]
            "context_text": context_text,
        }

        # 3) Nếu cung cấp encoder, head tự nạp context để trả context_plus
        if t5_encoder is not None:
            dev = h_fused.device
            with torch.set_grad_enabled(
                self.training and any(p.requires_grad for p in t5_encoder.parameters())
            ):
                enc_out = t5_encoder(
                    input_ids=extras["input_ids"].to(dev),
                    attention_mask=extras["attention_mask"].to(dev),
                ).last_hidden_state
            # Residual-bias + LN
            context_hidden = enc_out  # [1, S, D]
            h_ctx_hat = self.ln(self.do(context_hidden + b.to(dev)))  # [1, S, D]
            out["context_hidden"] = context_hidden
            out["context_plus"] = h_ctx_hat

        return out
