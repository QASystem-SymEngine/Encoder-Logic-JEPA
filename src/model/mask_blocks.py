# mask_blocks.py
from typing import Dict, List, Tuple
import random


def build_blocks_simple(item_mix: Dict) -> List[List[int]]:
    """
    Trả về danh sách block, mỗi block là list các chỉ số token i (global).
    Ở bản đơn giản: 1 block = 1 token meaningful (không ngoặc).
    """
    tokens: List[Tuple[str, int]] = item_mix["tokens"]
    blocks: List[List[int]] = []
    for i, (tok, _) in enumerate(tokens):
        if tok in {"(", ")"}:
            continue
        if (
            tok == "<extra_id_0>"
        ):  # nếu bạn có cắm sep như vậy trong tokens (thường không)
            continue
        blocks.append([i])
    return blocks


def sample_mask_flags(
    item_mix: Dict, mask_ratio: float = 0.2, seed: int | None = None
) -> List[int]:
    rng = random.Random(seed)
    blocks = build_blocks_simple(item_mix)
    n_mask = max(1, int(round(len(blocks) * mask_ratio)))
    chosen = set(
        idx for blk in rng.sample(blocks, k=min(n_mask, len(blocks))) for idx in blk
    )
    L = len(item_mix["tokens"])
    mask_flags = [1 if i in chosen else 0 for i in range(L)]
    # bracket luôn 0
    for i, (tok, _) in enumerate(item_mix["tokens"]):
        if tok in {"(", ")"}:
            mask_flags[i] = 0
    return mask_flags
