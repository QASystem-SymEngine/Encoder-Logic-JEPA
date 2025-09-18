# =============================
# src/model.py
# =============================
from __future__ import annotations
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from typing import List, Optional, Tuple
from typing import Dict, Any
import os
import torch

def load_tokenizer(
    model_name: str,
) -> Tuple[T5TokenizerFast, T5ForConditionalGeneration]:
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    return tokenizer


def load_pretrained_encoder_weights(
    model: T5ForConditionalGeneration, encoder_state_path: str
) -> tuple[list, list]:
    state = torch.load(encoder_state_path, map_location="cpu")
    missing_keys, unexpected_keys = model.encoder.load_state_dict(state, strict=False)
    return missing_keys, unexpected_keys


def freeze_encoder(model: T5ForConditionalGeneration) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False

def save_best_model(model_dict: dict, save_dir: str, tag: str):
    """
    Lưu state_dict của từng thành phần trong JEPA.
    tag: 'epoch3' hoặc 'best'
    """
    os.makedirs(save_dir, exist_ok=True)
    for name, module in model_dict.items():
        if hasattr(module, "encoder"):
            state_dict = module.encoder.state_dict()
        else:
            state_dict = module.state_dict()

        save_path = os.path.join(save_dir, f"{name}_{tag}.pth")
        torch.save(state_dict, save_path)
        print(f"[Info] Saved {name} -> {save_path}")

class ASTEncoder:
    def __init__(
        self,
        t5_name: str = "t5-base",
        device: str | torch.device = None,
        agg: str = "mean",
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.agg = agg

        # chỉ load tokenizer và embedding matrix
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_name)
        model = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.embedding_matrix = model.get_input_embeddings().weight.to(device)
        del model
        torch.cuda.empty_cache()

        self.cache_tokenstr_to_emb: Dict[str, torch.Tensor] = {}

    # -------------------------
    # Helpers: build token embeddings từ static embedding matrix
    # -------------------------
    def build_token_embedding_from_t5(
        self,
        token_str: str,
        tokenizer: T5TokenizerFast,
        embedding_matrix: torch.Tensor,
        device: torch.device,
        agg: str = "mean",
    ) -> torch.Tensor:
        """
        Return a CPU tensor embedding for token_str using static embedding matrix.
        - tokenizer.encode(..., add_special_tokens=False) -> list of subtoken ids
        - aggregate subtoken embeddings by mean (or sum)
        """
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 0:
            d = embedding_matrix.size(1)
            return torch.zeros(d, dtype=torch.float32)

        ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        with torch.no_grad():
            sub_embs = embedding_matrix[ids_tensor]  # (n_subtokens, d)
            if agg == "mean":
                out = sub_embs.mean(dim=0).detach().cpu()
            elif agg == "sum":
                out = sub_embs.sum(dim=0).detach().cpu()
            else:
                raise ValueError("agg must be 'mean' or 'sum'")
        return out

    # -------------------------
    # Encode using only value_paths
    # -------------------------
    def encode_from_json_with_token_init(
        self, data: Dict[str, Any], token_init_embeddings: Dict[int, torch.Tensor]
    ):
        dim = (
            next(iter(token_init_embeddings.values())).shape[0]
            if token_init_embeddings
            else 0
        )

        def emb_of_id(tid: int) -> torch.Tensor:
            return token_init_embeddings.get(tid, torch.zeros(dim, dtype=torch.float32))

        node_embeddings: Dict[int, torch.Tensor] = {}
        node_name_map: Dict[int, str] = {}

        for entry in data.get("value_paths", []):
            current_id = int(entry["current_id"])
            current_node = entry.get("current_node", "")
            path_ids = [int(x) for x in entry.get("path_ids", [])]
            if not path_ids:
                continue
            emb_stack = torch.stack([emb_of_id(tid) for tid in path_ids], dim=0)
            path_vec = emb_stack.mean(dim=0)
            node_embeddings[current_id] = path_vec
            node_name_map[current_id] = current_node

        return {"node_embeddings": node_embeddings, "node_name_map": node_name_map}

    def process_one_2(self, data: dict):
        """Xử lý 1 dòng json."""
        token_init_embeddings: Dict[int, torch.Tensor] = {}
        for tok_str, tid in data.get("tokens", []):
            tid = int(tid)
            if tok_str in self.cache_tokenstr_to_emb:
                token_init_embeddings[tid] = self.cache_tokenstr_to_emb[tok_str]
            else:
                emb = self.build_token_embedding_from_t5(
                    tok_str,
                    self.tokenizer,
                    self.embedding_matrix,
                    self.device,
                    agg=self.agg,
                )
                self.cache_tokenstr_to_emb[tok_str] = emb
                token_init_embeddings[tid] = emb

        out = self.encode_from_json_with_token_init(data, token_init_embeddings)
        out["token_init_embeddings"] = token_init_embeddings
        out["tokens_map"] = {int(tid): tok for tok, tid in data.get("tokens", [])}
        return out

    def process_one(self, data: dict):
        """Xử lý 1 dòng json."""
        token_init_embeddings: Dict[int, torch.Tensor] = {}

        # data["tokens"] là list[[tok_str, tid], ...]
        for tok_str, tid in data["tokens"]:
            tid = int(tid)
            if tok_str in self.cache_tokenstr_to_emb:
                token_init_embeddings[tid] = self.cache_tokenstr_to_emb[tok_str]
            else:
                emb = self.build_token_embedding_from_t5(
                    tok_str,
                    self.tokenizer,
                    self.embedding_matrix,
                    self.device,
                    agg=self.agg,
                )
                self.cache_tokenstr_to_emb[tok_str] = emb
                token_init_embeddings[tid] = emb

        out = self.encode_from_json_with_token_init(data, token_init_embeddings)
        out["token_init_embeddings"] = token_init_embeddings

        # xây map id -> token
        out["tokens_map"] = {int(tid): tok for tok, tid in data["tokens"]}
        return out

    def process_batch(self, batch):
        """Xử lý 1 batch data (list[dict])."""
        return [self.process_one(d) for d in batch]

    def process_ast_nl_batch(self, ast_nl_batch):
        """
        ast_nl_batch: list[list[dict]], 
            mỗi phần tử là 1 topic_ast = list các dict (mỗi dict là 1 expression trong topic).
        Trả về list[list], mỗi topic -> list kết quả ASTEncoder cho từng expression
        """
        all_outputs = []
        for topic_ast in ast_nl_batch:  # topic_ast = list[dict]
            topic_outputs = []
            for expr in topic_ast:  # expr = dict của 1 expression
                out = self.process_one(expr)
                topic_outputs.append(out)
            all_outputs.append(topic_outputs)
        return all_outputs

    def build_sentence_and_spans(
        self, tokens: List[Tuple[str, int]], tokenizer: T5TokenizerFast
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        tokens: list of (token_str, token_id) in the expression order.
        Returns:
        - ids_all: list of subtoken ids (no special tokens)
        - spans: list of (start, end) per original token (end exclusive)
        """
        ids_all: List[int] = []
        spans: List[Tuple[int, int]] = []
        for tok_str, _ in tokens:
            sub_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            start = len(ids_all)
            ids_all.extend(sub_ids)
            end = len(ids_all)
            spans.append((start, end))
        return ids_all, spans

    def inject_ast_embeddings(
        self,
        batch: dict,
        ast_batch_out: list[dict],
        context_hidden: torch.Tensor,
        input_context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enrich context_hidden bằng AST embeddings (saved_token_init_embeddings).
        - batch: dict, chứa batch["ast_fol"]
        - ast_batch_out: list kết quả từ ASTEncoder.process_batch
        - context_hidden: (B, L, d)
        - input_context_ids: (B, L)
        Return: context_hidden đã cộng thêm AST vec vào đúng span.
        """
        B, L, d = context_hidden.size()
        device = context_hidden.device

        for b_idx, ast_out in enumerate(ast_batch_out):
            tokens_list = [
                (tok, int(tid)) for tok, tid in batch["ast_fol"][b_idx]["tokens"]
            ]

            # build spans từ tokens_list
            ids_all, spans = self.build_sentence_and_spans(tokens_list, self.tokenizer)

            # map token -> subtoken pieces (ở đây ta không cần run encoder lại, vì context_hidden đã có)
            token_to_pieces: dict[int, list[int]] = {}
            for idx, (_, tid) in enumerate(tokens_list):
                s, e = spans[idx]
                token_to_pieces[int(tid)] = list(range(s, e))

            saved_token_inits: dict[int, torch.Tensor] = ast_out.get(
                "token_init_embeddings", {}
            )

            # cộng vector đã lưu vào context_hidden ở vị trí span
            for tid, span_idxs in token_to_pieces.items():
                if not span_idxs:
                    continue
                saved_vec = saved_token_inits.get(tid)
                if saved_vec is None:
                    continue
                saved_vec = saved_vec.to(device)
                for pos in span_idxs:
                    if pos < L:  # tránh out-of-bound
                        context_hidden[b_idx, pos] += saved_vec
        return context_hidden

    def _find_subsequence(
        self, sequence: List[int], subseq: List[int], start: int = 0
    ) -> int:
        """Tìm vị trí đầu tiên của subseq trong sequence, bắt đầu từ start."""
        if not subseq:
            return -1
        n = len(sequence)
        m = len(subseq)
        for i in range(start, n - m + 1):
            if sequence[i : i + m] == subseq:
                return i
        return -1

    def build_sentence_and_spans_sequential(
        self, nl: str, tokens: List[Tuple[str, int]], tokenizer: T5TokenizerFast
    ) -> Tuple[List[int], List[Optional[Tuple[int, int]]]]:
        """
        Mã hóa câu NL và tìm span của các token theo thứ tự từ trái sang phải.
        Trả về:
        - ids_all: Danh sách ID subtoken của câu.
        - spans: Danh sách (start, end) hoặc None nếu token không khớp.
        """
        ids_all = tokenizer.encode(nl, add_special_tokens=False)
        spans: List[Optional[Tuple[int, int]]] = []
        search_start = 0
        for tok_str, _ in tokens:
            sub_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(sub_ids) == 0:
                spans.append(None)
                continue
            pos = self._find_subsequence(ids_all, sub_ids, start=search_start)
            if pos >= 0:
                spans.append((pos, pos + len(sub_ids)))
                search_start = pos + len(sub_ids)
            else:
                spans.append(None)
        return ids_all, spans

    def inject_ast_embeddings_nl(
        self,
        batch: dict,
        ast_nl_batch_out: list[list[dict]],
        context_hidden: torch.Tensor,
        input_context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Thêm vector AST vào context_hidden cho các token khớp với các câu NL trong ast_nl.
        - batch["ast_nl"]: list[list[dict]], mỗi sample có nhiều câu (mỗi dict là một câu).
        - ast_nl_batch_out: list[list[dict]], kết quả từ process_ast_nl_batch cho mỗi câu.
        - context_hidden: (B, L, d) tensor biểu diễn ẩn.
        - input_context_ids: (B, L) tensor chứa ID token.
        Trả về: context_hidden được cập nhật với vector AST tại các span khớp.
        """
        B, L, d = context_hidden.size()
        device = context_hidden.device

        for b_idx in range(B):
            ast_out_list = ast_nl_batch_out[b_idx]  # list[dict] cho sample b_idx
            ast_nl_list = batch["ast_nl"][b_idx]  # list[dict] cho sample b_idx

            search_start = 0  # vị trí offset trong đoạn văn

            # duyệt song song qua từng câu
            for ast_out, nl_item in zip(ast_out_list, ast_nl_list):
                nl_text = nl_item["expression"].strip()
                tokens_list = [(tok, int(tid)) for tok, tid in nl_item["tokens"]]

                # Xây dựng danh sách ID và span cho câu hiện tại
                ids_all, spans = self.build_sentence_and_spans_sequential(
                    nl_text, tokens_list, self.tokenizer
                )

                # Ánh xạ token ID tới các vị trí subtoken trong đoạn văn
                token_to_pieces: Dict[int, List[int]] = {}
                for idx, (_, tid) in enumerate(tokens_list):
                    span = spans[idx]
                    if span is None:
                        continue
                    s, e = span
                    token_to_pieces[tid] = list(
                        range(s + search_start, e + search_start)
                    )

                # Lấy vector AST đã lưu
                saved_token_inits: Dict[int, torch.Tensor] = ast_out.get(
                    "token_init_embeddings", {}
                )

                # Cộng vector AST vào context_hidden tại các span khớp
                for tid, span_idxs in token_to_pieces.items():
                    if not span_idxs:
                        continue
                    saved_vec = saved_token_inits.get(tid)
                    if saved_vec is None:
                        continue
                    saved_vec = saved_vec.to(device=device, dtype=context_hidden.dtype)
                    for pos in span_idxs:
                        if pos < L:
                            context_hidden[b_idx, pos] += saved_vec

                # Cập nhật search_start cho câu tiếp theo (+1 cho dấu chấm phân tách)
                if ids_all:
                    search_start += len(ids_all) + 1

        return context_hidden
