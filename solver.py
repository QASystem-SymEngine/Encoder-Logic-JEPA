import math
import time
import torch
import os
from models import *
from data import *
from utils import *
from jepa import *
from transformers import get_cosine_schedule_with_warmup

import wandb


class Solver:
    def __init__(self, paths, cfg):
        self.paths = paths
        self.cfg = cfg

        self.model_dir = make_save_dir(paths.model_dir)
        self.device = cfg.device

        self.train_data = ProcessData(paths, cfg, mode="train")  # truyền cả paths & cfg
        self.val_data = ProcessData(paths, cfg, mode="val")

        # Tự tính steps_per_epoch
        self.steps_per_epoch = math.ceil(len(self.train_data) / cfg.batch_size)
        self.total_steps = cfg.num_epochs * self.steps_per_epoch
        print(
            f"Total steps: {self.total_steps}, Steps per epoch: {self.steps_per_epoch}"
        )

        # 2) Tokenizer & model
        self.tokenizer = load_tokenizer(cfg.model_name)

        self.ast_encoder = ASTEncoder(t5_name="t5-base", device=cfg.device, agg="mean")

        # ---- JEPA Modules ----
        def build_component(cls, name: str):
            """
            cls: class của module (JEPAEncoder / JEPAPredictor)
            name: 'context_encoder', 'target_encoder', 'predictor'
            """
            if getattr(cfg, "is_load_state_dict", False):
                best_path = os.path.join(paths.model_dir, f"{name}_best.pth")
                if os.path.exists(best_path):
                    print(f"[Info] Loading {name} (best) from {best_path}")
                    return cls(model_name=cfg.model_name, encoder_state_path=best_path)

                print(
                    f"[Info] No checkpoint found, loading pretrained {cfg.model_name} instead."
                )
            return cls(model_name=getattr(cfg, "model_name", "t5-base"))

        # Context Encoder
        self.context_encoder = build_component(JEPAEncoder, "context_encoder")

        # Target Encoder
        self.target_encoder = build_component(JEPAEncoder, "target_encoder")

        # Predictor
        self.predictor = build_component(JEPAPredictor, "predictor")

        self.positional_embedding = nn.Embedding(512, 768)
        # Init với sinusoidal values
        init_sinusoidal_embeddings(self.positional_embedding)

        if cfg.model_name == "t5-base":
            self.hidden_size = 768
        elif cfg.model_name == "t5-large":
            self.hidden_size = 1024
        else:  # t5-small
            self.hidden_size = 512

        # Mask embedding (khởi tạo vector duy nhất)
        self.mask_embedding = nn.Parameter(
            torch.randn(self.hidden_size, device=self.device)
        )

        self.best_val_loss = float("inf")

        # Di chuyển mô hình sang GPU nếu không sử dụng CPU
        if cfg.device == "cuda" and torch.cuda.is_available():
            self.context_encoder = self.context_encoder.to(self.device)
            self.target_encoder = self.target_encoder.to(self.device)
            self.predictor = self.predictor.to(self.device)
            self.positional_embedding = self.positional_embedding.to(self.device)
            self.mask_embedding = self.mask_embedding.to(self.device)

    def train(self):
        total_params = sum(
            p.numel()
            for p in list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
            if p.requires_grad
        )
        print("Total trainable parameters:", total_params)

        optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.positional_embedding.parameters())
            + [self.mask_embedding],
            lr=self.cfg.lr,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * self.total_steps),
            num_training_steps=self.total_steps,
        )

        beta = self.cfg.ema_decay
        accumulation_steps = (
            8  # cộng dồn gradient, giả sử batch hiệu quả = batch_size * 8
        )

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "Logic-JEPA-Pretrain"),
            name=os.environ.get("WANDB_NAME", "Logic-JEPA-Pretrain-18-09"),
            config=vars(self.cfg),
        )

        for epoch in range(self.cfg.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.cfg.num_epochs} ===")
            data_yielder = self.train_data.batch_yielder()
            epoch_loss = []
            start = time.time()

            optimizer.zero_grad()

            for step in range(self.steps_per_epoch):
                self.context_encoder.train()
                self.target_encoder.eval()
                self.predictor.train()

                batch = next(data_yielder)

                # ... (tokenize + context_hidden như code bạn giữ nguyên) ...
                # 1) Lấy danh sách text từ batch
                nl_list = [sample["text"] for sample in batch["text_tokens"]]

                # 1a) Loại bỏ khoảng trắng trước <extra_id_0> -> sẽ improve data chỗ này lại sau.
                nl_list = [
                    text.replace(" <extra_id_0>", "<extra_id_0>") for text in nl_list
                ]

                # 2) Tokenize source (KHÔNG pad)
                src = self.tokenizer(
                    nl_list,
                    max_length=self.cfg.max_length,
                    add_special_tokens=False,
                    truncation=True,
                    padding="longest",  # pad tất cả về cùng chiều dài của câu dài nhất trong batch
                    return_tensors="pt",
                )

                input_context_ids = src.input_ids  # shape: [batch_size, seq_len]
                attention_mask = src.attention_mask  # shape: [batch_size, seq_len]

                # Lấy chiều dài sau khi tokenize
                seq_len = input_context_ids.size(1)

                # Pad context_mask về cùng chiều dài seq_len
                context_mask_list = []
                for item in batch["text_tokens"]:
                    cm = item["context_mask"]
                    if len(cm) < seq_len:
                        # thêm 0 ở cuối
                        cm = cm + [0] * (seq_len - len(cm))
                    else:
                        # cắt nếu dài hơn (đảm bảo cùng seq_len)
                        cm = cm[:seq_len]
                    context_mask_list.append(cm)

                context_mask = torch.tensor(
                    context_mask_list, device=input_context_ids.device
                )

                # 3) Di chuyển sang GPU nếu cần
                if self.cfg.device == "cuda" and torch.cuda.is_available():
                    input_context_ids = input_context_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    context_mask = context_mask.to(self.device)

                    # ---- FORWARD ----
                # 1. Forward context encoder (nhận representation từ câu context)
                context_hidden = self.context_encoder(
                    input_context_ids, attention_mask
                )  # (batch, seq_len, hidden_size)

                # Thay thế embedding tại các vị trí mask
                batch_size, seq_len, hidden_size = context_hidden.shape
                # Tạo tensor mask_embedding cho tất cả vị trí mask
                mask_positions = (
                    (context_mask == 0).float().unsqueeze(-1)
                )  # [batch_size, seq_len, 1]
                mask_embedding_expanded = (
                    self.mask_embedding.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, seq_len, hidden_size)
                )  # [batch_size, seq_len, hidden_size]

                # Thay thế tại các vị trí mask, giữ nguyên các vị trí không mask
                context_hidden_masked = (
                    context_hidden * (1 - mask_positions)
                    + mask_embedding_expanded * mask_positions
                )  # [batch_size, seq_len, hidden_size]

                # Thêm positional embedding cho toàn bộ chuỗi
                positions = (
                    torch.arange(seq_len)
                    .unsqueeze(0)
                    .expand(batch_size, seq_len)
                    .to(context_hidden_masked.device)
                )
                pos_emb = self.positional_embedding(
                    positions
                )  # [batch_size, seq_len, hidden_size]
                context_hidden_with_pos = (
                    context_hidden_masked + pos_emb
                )  # [batch_size, seq_len, hidden_size]

                # 4) Encode AST (batch["ast_fol"] là list[dict])
                ast_fol_out = self.ast_encoder.process_batch(batch["ast_fol"])
                # ast_batch_out là list[dict] với key: node_embeddings, node_name_map, token_init_embeddings, tokens_map

                # Encode AST for NL
                ast_nl_out = self.ast_encoder.process_ast_nl_batch(batch["ast_nl"])

                # === Inject AST với scale nhỏ hơn ===
                context_hidden_with_ast_fol = self.ast_encoder.inject_ast_embeddings(
                    batch, ast_fol_out, context_hidden_with_pos, input_context_ids
                )
                context_hidden_with_ast_fol *= 0.8  # scale giảm nhẹ

                context_hidden_with_ast_nl = self.ast_encoder.inject_ast_embeddings_nl(
                    batch, ast_nl_out, context_hidden_with_ast_fol, input_context_ids
                )
                context_hidden_with_ast_nl *= 0.8

                predicted_rep = self.predictor(
                    context_hidden_with_ast_nl, attention_mask
                )

                with torch.no_grad():
                    target_rep = self.target_encoder(
                        input_context_ids, attention_mask
                    ).detach()

                mask_indices = (
                    (context_mask == 0).unsqueeze(-1).expand_as(predicted_rep)
                )
                masked_pred = predicted_rep[mask_indices].view(-1, self.hidden_size)
                masked_target = target_rep[mask_indices].view(-1, self.hidden_size)

                mse_loss = nn.MSELoss(reduction="mean")
                loss = mse_loss(masked_pred, masked_target)

                # === Gradient accumulation ===
                loss = loss / accumulation_steps
                loss.backward()

                # Clip grad để tránh nổ
                torch.nn.utils.clip_grad_norm_(
                    list(self.context_encoder.parameters())
                    + list(self.predictor.parameters())
                    + list(self.positional_embedding.parameters())
                    + [self.mask_embedding],
                    max_norm=1.0,
                )

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # EMA update target encoder
                    for param_q, param_k in zip(
                        self.context_encoder.parameters(),
                        self.target_encoder.parameters(),
                    ):
                        param_k.data = beta * param_k.data + (1.0 - beta) * param_q.data

                epoch_loss.append(loss.item() * accumulation_steps)  # scale lại để log

                if step % self.cfg.log_every == 0:
                    print(
                        f"[Epoch {epoch + 1} | Step {step}] Step Loss: {loss.item() * accumulation_steps:.4f}"
                    )
                    wandb.log(
                        {
                            "step_loss": loss.item() * accumulation_steps,
                            "epoch": epoch + 1,
                            "step": step + epoch * self.steps_per_epoch,
                        }
                    )

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            print(
                f"--> Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss:.4f}. Time: {time.time() - start:.2f}s"
            )

            # chạy validation
            val_loss = self.evaluate(epoch)

            wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1})
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_best_model(
                    {
                        "context_encoder": self.context_encoder,
                        "target_encoder": self.target_encoder,
                        "predictor": self.predictor,
                    },
                    save_dir=self.paths.model_dir,
                    tag=f"best",
                )
        wandb.finish()
        print("Training completed.")

    @torch.no_grad()
    def evaluate(self, epoch):
        print(f"--> Running validation at epoch {epoch+1}")
        self.context_encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()

        data_yielder = self.val_data.batch_yielder()
        val_losses = []

        for step, batch in enumerate(data_yielder):
            # 1) Lấy text từ batch
            nl_list = [sample["text"] for sample in batch["text_tokens"]]
            nl_list = [
                text.replace(" <extra_id_0>", "<extra_id_0>") for text in nl_list
            ]

            src = self.tokenizer(
                nl_list,
                max_length=self.cfg.max_length,
                add_special_tokens=False,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            input_context_ids = src.input_ids
            attention_mask = src.attention_mask
            seq_len = input_context_ids.size(1)

            # Pad context_mask
            context_mask_list = []
            for item in batch["text_tokens"]:
                cm = item["context_mask"]
                if len(cm) < seq_len:
                    cm = cm + [0] * (seq_len - len(cm))
                else:
                    cm = cm[:seq_len]
                context_mask_list.append(cm)

            context_mask = torch.tensor(
                context_mask_list, device=input_context_ids.device
            )

            if self.cfg.device == "cuda" and torch.cuda.is_available():
                input_context_ids = input_context_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                context_mask = context_mask.to(self.device)

            # ---- Forward ----
            context_hidden = self.context_encoder(input_context_ids, attention_mask)

            batch_size, seq_len, hidden_size = context_hidden.shape
            mask_positions = (context_mask == 0).float().unsqueeze(-1)
            mask_embedding_expanded = (
                self.mask_embedding.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, seq_len, hidden_size)
            )
            context_hidden_masked = (
                context_hidden * (1 - mask_positions)
                + mask_embedding_expanded * mask_positions
            )

            positions = (
                torch.arange(seq_len)
                .unsqueeze(0)
                .expand(batch_size, seq_len)
                .to(context_hidden_masked.device)
            )
            pos_emb = self.positional_embedding(positions)
            context_hidden_with_pos = context_hidden_masked + pos_emb

            ast_fol_out = self.ast_encoder.process_batch(batch["ast_fol"])
            ast_nl_out = self.ast_encoder.process_ast_nl_batch(batch["ast_nl"])
            context_hidden_with_ast_fol = self.ast_encoder.inject_ast_embeddings(
                batch, ast_fol_out, context_hidden_with_pos, input_context_ids
            )
            context_hidden_with_ast_nl = self.ast_encoder.inject_ast_embeddings_nl(
                batch, ast_nl_out, context_hidden_with_ast_fol, input_context_ids
            )

            predicted_rep = self.predictor(context_hidden_with_ast_nl, attention_mask)

            target_rep = self.target_encoder(input_context_ids, attention_mask).detach()

            mask_indices = (context_mask == 0).unsqueeze(-1).expand_as(predicted_rep)
            masked_pred = predicted_rep[mask_indices].view(-1, hidden_size)
            masked_target = target_rep[mask_indices].view(-1, hidden_size)

            mse_loss = nn.MSELoss(reduction="mean")
            loss = mse_loss(masked_pred, masked_target)
            val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
        return avg_val_loss
