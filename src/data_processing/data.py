from __future__ import print_function
from src.utils.util import *


class ProcessData:
    def __init__(self, paths, cfg, mode="train"):
        self.batch_size = cfg.batch_size
        self.mode = mode

        # Load dữ liệu
        if mode == "train":
            self.data = load_jsonl_dataset(paths.train_path)
        elif mode == "val":
            self.data = load_jsonl_dataset(paths.val_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        """Nếu cần tokenizer, implement ở đây."""
        return [ord(c) for c in text]  # placeholder, thay bằng tokenizer thật

    def batch_yielder(self):
        """Hàm chung cho train/val/test"""
        batch = {"topic": [], "ast_nl": [], "ast_fol": []}
        batch_size = self.batch_size

        for sample in self.data:
            topic = sample.get("topic")
            ast_nl = sample.get("ast_nl", [])
            ast_fol = sample.get("ast_fol", [])

            if isinstance(ast_nl, dict):
                ast_nl = [ast_nl]

            if (
                isinstance(ast_fol, list)
                and len(ast_fol) == 1
                and isinstance(ast_fol[0], dict)
            ):
                ast_fol = ast_fol[0]

            batch["topic"].append(topic)
            batch["ast_nl"].append(ast_nl)
            batch["ast_fol"].append(ast_fol)

            if len(batch["ast_fol"]) >= batch_size:
                yield {
                    "topic": batch["topic"],
                    "ast_nl": batch["ast_nl"],
                    "ast_fol": batch["ast_fol"],
                }
                batch = {"topic": [], "ast_nl": [], "ast_fol": []}

        if batch["ast_fol"]:
            yield batch
