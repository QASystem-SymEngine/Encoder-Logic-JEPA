# mixed_linearizer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

Modality = Literal["NL", "FOL"]


@dataclass
class Segment:
    seg_id: int
    modality: Modality  # "NL" | "FOL"
    expression: str
    tokens: List[Tuple[str, int]]  # (string, local_id)
    type_paths: List[Dict] | None  # FOL có thể có, NL thường None
    value_paths: List[Dict] | None


def _shift_paths(
    paths: List[Dict] | None, id_map_local2global: Dict[int, int]
) -> List[Dict]:
    if not paths:
        return []
    out = []
    for p in paths:
        g = dict(p)
        g["current_id"] = id_map_local2global.get(p["current_id"], p["current_id"])
        # Giữ nguyên "paths" (chuỗi) và "path_ids" (số) như dữ liệu gốc
        out.append(g)
    return out


def linearize_sample(sample_data: Dict) -> Dict:
    """
    Nhập `sample_data` có keys: ast_nl: List[...], ast_fol: List[...]
    Trả về `item_mix` chuẩn hoá duy nhất:
      - tokens: List[(tok, global_id)]
      - seg_meta: List[(seg_id, modality_int)] với modality_int: 0=NL, 1=FOL
      - type_paths, value_paths: đã shift current_id -> global_id
      - expressions_cat: NL câu1 ⟂ NL câu2 ⟂ ... ⟂ FOL1 (ghép để encode context)
    """
    segments: List[Segment] = []
    seg_id = 0

    # NL trước
    for nl in sample_data.get("ast_nl", []):
        segments.append(
            Segment(
                seg_id=seg_id,
                modality="NL",
                expression=nl["expression"],
                tokens=[(tok, lid) for tok, lid in nl["tokens"]],
                type_paths=None,  # NL không có type_paths
                value_paths=nl.get("value_paths", []),
            )
        )
        seg_id += 1

    # FOL sau
    for fol in sample_data.get("ast_fol", []):
        segments.append(
            Segment(
                seg_id=seg_id,
                modality="FOL",
                expression=fol["expression"],
                tokens=[(tok, lid) for tok, lid in fol["tokens"]],
                type_paths=fol.get("type_paths", []),
                value_paths=fol.get("value_paths", []),
            )
        )
        seg_id += 1

    # Gộp token + xây map local->global cho từng segment
    tokens: List[Tuple[str, int]] = []
    seg_meta: List[Tuple[int, int]] = []  # (seg_id, modality_int)
    type_paths_all: List[Dict] = []
    value_paths_all: List[Dict] = []

    gid = 0
    expressions = []
    for seg in segments:
        # map local id -> global id theo thứ tự xuất hiện
        id_map = {}
        for tok, lid in seg.tokens:
            id_map[lid] = gid
            tokens.append((tok, gid))
            seg_meta.append((seg.seg_id, 0 if seg.modality == "NL" else 1))
            gid += 1
        # shift paths current_id sang global id
        type_paths_all += _shift_paths(seg.type_paths, id_map)
        value_paths_all += _shift_paths(seg.value_paths, id_map)
        expressions.append(seg.expression)

    expressions_cat = " <extra_id_0> ".join(expressions)  # phân tách nhẹ nhàng

    return {
        "topic": sample_data.get("topic"),
        "tokens": tokens,  # [(tok, global_id)]
        "seg_meta": seg_meta,  # [(seg_id, modality_int)]
        "type_paths": type_paths_all,  # current_id đã là global
        "value_paths": value_paths_all,  # current_id đã là global
        "expressions_cat": expressions_cat,  # để encode context
        "num_segments": len(segments),
    }
