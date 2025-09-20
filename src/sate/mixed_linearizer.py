# mixed_linearizer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Any

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
        out.append(g)
    return out


def _as_list(x: Any) -> List[Dict]:
    """Chuẩn hoá trường có thể là None | dict | list(dict) thành list(dict)."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # x là một item (dict, hoặc hiếm khi kiểu khác) -> bọc lại thành list
    return [x]


def _norm_tokens(tok_list: Any) -> List[Tuple[str, int]]:
    """Bảo thủ: nhận list[[tok, id] | (tok, id)] -> list[(str, int)]."""
    out: List[Tuple[str, int]] = []
    if not tok_list:
        return out
    for t in tok_list:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            tok, lid = t
            out.append((str(tok), int(lid)))
    return out


def linearize_sample(sample_data: Dict) -> Dict:
    """
    Input:
      - ast_nl: List[dict] hoặc None
      - ast_fol: dict (một item) hoặc List[dict] hoặc None
    Output item_mix chuẩn hoá:
      - tokens: List[(tok, global_id)]
      - seg_meta: List[(seg_id, modality_int)] với modality_int: 0=NL, 1=FOL
      - type_paths, value_paths: đã shift current_id -> global_id
      - expressions_cat: NL câu1 ⟂ ... ⟂ FOL1
    """
    segments: List[Segment] = []
    seg_id = 0

    # NL (đã chuẩn hoá về list)
    for nl in _as_list(sample_data.get("ast_nl")):
        segments.append(
            Segment(
                seg_id=seg_id,
                modality="NL",
                expression=nl["expression"],
                tokens=_norm_tokens(nl.get("tokens", [])),
                type_paths=None,
                value_paths=nl.get("value_paths", []),
            )
        )
        seg_id += 1

    # FOL (chịu được ast_fol là 1 item hoặc list)
    for fol in _as_list(sample_data.get("ast_fol")):
        segments.append(
            Segment(
                seg_id=seg_id,
                modality="FOL",
                expression=fol["expression"],
                tokens=_norm_tokens(fol.get("tokens", [])),
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
    expressions: List[str] = []
    for seg in segments:
        id_map: Dict[int, int] = {}
        for tok, lid in seg.tokens:
            id_map[lid] = gid
            tokens.append((tok, gid))
            seg_meta.append((seg.seg_id, 0 if seg.modality == "NL" else 1))
            gid += 1
        type_paths_all += _shift_paths(seg.type_paths, id_map)
        value_paths_all += _shift_paths(seg.value_paths, id_map)
        expressions.append(seg.expression)

    expressions_cat = " <extra_id_0> ".join(expressions)

    return {
        "topic": sample_data.get("topic"),
        "tokens": tokens,
        "seg_meta": seg_meta,
        "type_paths": type_paths_all,
        "value_paths": value_paths_all,
        "expressions_cat": expressions_cat,
        "num_segments": len(segments),
    }
