import math
import os
import time
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import AdamW

from src.sate.mixed_linearizer import linearize_sample
from src.model.models import load_t5_model, load_tokenizer
from src.data_processing.data import ProcessData
from src.model.jepa_mixed import MixedNLFOLJEPA
from src.model.mask_blocks import sample_mask_flags
from src.sate.fusion_gating import FusionConfig, SATEWithFusion
from src.sate.sate_feature_encoder import SATEFeatureEncoder
from src.sate.structural_linearized_stream import (
    StructuralTransformer,
    build_mixed_bias,
)
from src.utils.util import set_seed, count_params, make_save_dir
import wandb


class Solver:
    def __init__(self, paths, cfg):
        self.paths = paths
        self.cfg = cfg

        # Tạo folder lưu model
        self.model_dir = make_save_dir(paths.model_dir)
        self.device = torch.device(cfg.device)

        # Load data
        self.train_data = ProcessData(paths, cfg, mode="train")
        self.val_data = ProcessData(paths, cfg, mode="val")

        # Tokenizer
        self.tokenizer = load_tokenizer(cfg.model_name)
        self.t5_model = load_t5_model(cfg.model_name)
        if not cfg.fine_tune_t5:
            for p in self.t5_model.encoder.parameters():
                p.requires_grad = False

        # Best val loss
        self.best_val_loss = float("inf")

        # WandB
        if cfg.use_wandb:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "Logic-JEPA-Pretrain"),
                name=os.environ.get("WANDB_NAME", "Logic-JEPA-Run"),
                config=vars(self.cfg),
            )

        # -------- Build configs từ cfg --------
        self.sate_cfg = cfg.sate_cfg

        self.struct_cfg = cfg.struct_cfg

        self.fusion_cfg = FusionConfig(
            per_dim_alpha=cfg.per_dim_alpha,
            dropout=cfg.fusion_dropout,
        )

    # =====================================================
    def build_model(self):
        """Khởi tạo encoder + JEPA model"""
        sate = SATEFeatureEncoder(
            self.sate_cfg, tokenizer=self.tokenizer, t5_model=self.t5_model
        ).to(self.device)
        struct = StructuralTransformer(self.struct_cfg).to(self.device)

        fusion = SATEWithFusion(
            sate_encoder=sate,
            structural_transformer=struct,
            fusion_cfg=self.fusion_cfg,
            bias_builder=build_mixed_bias,
        ).to(self.device)

        jepa = MixedNLFOLJEPA(
            online_encoder=fusion,
            target_encoder=deepcopy(fusion),
            d_model=self.sate_cfg.d_ast,
            loss_type="mse",
            use_bias_stats=True,
            lambda_sa=self.cfg.lambda_sa,
            lambda_ds=self.cfg.lambda_ds,
            lambda_gr=self.cfg.lambda_gr,
            lambda_xl=self.cfg.lambda_xl,
        ).to(self.device)

        print(f"[Info] Trainable params: {count_params(jepa):,}")
        return jepa

    # =====================================================
    def train(self):
        set_seed(1234)

        jepa = self.build_model()

        optim = AdamW(
            params=[p for p in jepa.parameters() if p.requires_grad],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        grad_clip = self.cfg.grad_clip
        ema_m = self.cfg.ema_decay

        data_yielder = self.train_data.batch_yielder()

        self.steps_per_epoch = math.ceil(len(self.train_data) / self.cfg.batch_size)

        global_step = 0
        best_loss = float("inf")

        for epoch in range(1, self.cfg.num_epochs + 1):
            t0 = time.time()
            running = 0.0
            for step in range(self.steps_per_epoch):
                try:
                    batch = next(
                        data_yielder
                    )  # batch là dict: {"topic": [...], "ast_nl": [...], "ast_fol": [...]}
                except StopIteration:
                    # Hết data sớm hơn dự kiến → khởi tạo lại yielder
                    data_yielder = self.train_data.batch_yielder()
                    batch = next(data_yielder)

                # ---- Chuẩn hoá batch → list các item theo sample ----
                bs = len(batch["topic"])
                items = [
                    {
                        "topic": batch["topic"][i],
                        "ast_nl": batch["ast_nl"][i],
                        "ast_fol": batch["ast_fol"][i],
                    }
                    for i in range(bs)
                ]

                # ---- Train trên từng item trong batch ----
                batch_loss = 0.0
                for item in items:
                    item_mix = linearize_sample(item)
                    mask_flags = sample_mask_flags(
                        item_mix, mask_ratio=0.20, seed=epoch * 1000 + step + 1
                    )
                    jepa.train()
                    optim.zero_grad(set_to_none=True)
                    out = jepa(item_full=item_mix, mask_flags=mask_flags, ema_m=ema_m)
                    loss = out["loss"]

                    if not torch.isfinite(loss):
                        print(f"[Warn] Non-finite loss at step {step+1}")
                        continue

                    loss.backward()
                    nn.utils.clip_grad_norm_(jepa.parameters(), max_norm=grad_clip)
                    optim.step()
                    batch_loss += loss.item()

                running += batch_loss / max(bs, 1)
                global_step += 1

                # Logging
                if (step + 1) % self.cfg.log_every == 0:
                    avg = running / self.cfg.log_every
                    extra = []
                    for k in ("loss_main", "loss_ds", "loss_gr", "loss_xl", "loss_sa"):
                        if k in out:
                            extra.append(f"{k}={out[k]:.6f}")
                    extra_str = (" | " + " ".join(extra)) if extra else ""
                    print(
                        f"[Epoch {epoch} | step {step+1:03d}] loss={avg:.6f}{extra_str}"
                    )

                    if self.cfg.use_wandb:
                        log_data = {"train/loss": avg, "global_step": global_step}
                        for k in (
                            "loss_main",
                            "loss_ds",
                            "loss_gr",
                            "loss_xl",
                            "loss_sa",
                        ):
                            if k in out:
                                log_data[f"train/{k}"] = out[k]
                        wandb.log(log_data, step=global_step)

                    running = 0.0

            # =============== Validation =================
            val_loss = self.validate(jepa, ema_m)
            epoch_time = time.time() - t0
            print(f"[Epoch {epoch} done in {epoch_time:.1f}s] val_loss={val_loss:.6f}")

            if self.cfg.use_wandb:
                wandb.log(
                    {"val/loss": val_loss, "epoch": epoch, "global_step": global_step},
                    step=global_step,
                )

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(
                    jepa, epoch, best_loss, optim=optim, global_step=global_step
                )
                self.save_t5_target_encoder(jepa, filename="t5_target_encoder.pth")
        print("[Done] Training finished.")

    # =====================================================
    def validate(self, jepa, ema_m):
        # dùng batch_yielder mặc định batch_size=cfg.batch_size
        val_yielder = self.val_data.batch_yielder()
        total_loss, n_items = 0.0, 0

        with torch.no_grad():
            jepa.eval()
            for vbatch in val_yielder:
                # vbatch: {"topic": [...], "ast_nl": [...], "ast_fol": [...]}
                bs = len(vbatch["topic"])
                items = [
                    {
                        "topic": vbatch["topic"][i],
                        "ast_nl": vbatch["ast_nl"][i],
                        "ast_fol": vbatch["ast_fol"][i],
                    }
                    for i in range(bs)
                ]

                for vitem in items:
                    vitem_mix = linearize_sample(vitem)  # phải linearize để có 'tokens'
                    mask_flags = sample_mask_flags(
                        vitem_mix, mask_ratio=0.20, seed=42 + n_items
                    )
                    out = jepa(item_full=vitem_mix, mask_flags=mask_flags, ema_m=ema_m)
                    total_loss += out["loss"].item()
                    n_items += 1

                # giới hạn nhanh để rút gọn thời gian validate
                if n_items >= 32:
                    break

        return total_loss / max(n_items, 1)

    # =====================================================
    def save_model(self, jepa, epoch, best_loss, optim=None, global_step=None):
        """
        Lưu đầy đủ:
        - online encoder
        - target encoder (EMA)
        - predictor
        - hint_proj, hint_alpha
        - optimizer state (nếu truyền vào)
        - thông tin huấn luyện: epoch, best_loss, global_step, cfg
        - RNG state để có thể resume deterministically
        """
        ckpt = {
            "online_encoder": jepa.online.state_dict(),
            "target_encoder": jepa.target.state_dict(),  # <-- thêm target
            "predictor": jepa.predictor.state_dict(),
            "hint_proj": jepa.hint_proj.state_dict(),
            "hint_alpha": jepa.hint_alpha.detach().cpu(),
            "epoch": epoch,
            "best_loss": best_loss,
            "global_step": global_step,
            "cfg": vars(self.cfg),  # tiện resume cấu hình
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        if optim is not None:
            ckpt["optimizer"] = optim.state_dict()

        save_path = os.path.join(self.model_dir, "mixed_jepa_best.pt")
        torch.save(ckpt, save_path)
        print(f"[Info] Saved best model → {save_path} (loss={best_loss:.6f})")

    # =====================================================
    def save_t5_target_encoder(self, jepa, filename="t5_target_encoder.pth"):
        """
        Lưu đúng T5 ENCODER thuộc khối TARGET (EMA).
        Đường dẫn đúng: jepa.target.sate.t5
        """
        t5 = jepa.target.sate.t5
        out_path = os.path.join(self.model_dir, filename)
        torch.save(t5.state_dict(), out_path)
        print(f"[Info] Saved target T5 encoder → {out_path}")
