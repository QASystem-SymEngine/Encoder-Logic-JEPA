import argparse
from types import SimpleNamespace

import wandb
from src.solver import Solver
from src.sate.sate_feature_encoder import SATEConfig
from src.sate.structural_linearized_stream import StructuralTransformerConfig


def parse_args():
    p = argparse.ArgumentParser()

    # ================= Data / IO =================
    p.add_argument("--train_path", default="./data/train_samples.jsonl")
    p.add_argument("--val_path", default="./data/train_samples.jsonl")
    p.add_argument("--data_dir", default="saved_data")
    p.add_argument("--model_dir", default="saved_models")

    # ================= Training =================
    p.add_argument("--train", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--log_every", type=int, default=20)

    # Optimizer & regularization
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # ================= Model SATEConfig =================
    p.add_argument("--model_name", type=str, default="t5-base")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--fine_tune_t5", type=bool, default=True)
    p.add_argument("--max_segments", type=int, default=10)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # ================= Structural Transformer =================
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_ff", type=int, default=2048)
    p.add_argument("--struct_dropout", type=float, default=0.1)
    p.add_argument("--bias_scale_init", type=float, default=1.0)
    p.add_argument("--bias_channels", type=int, default=4)

    # ================= Fusion Gating =================
    p.add_argument("--per_dim_alpha", type=bool, default=True)
    p.add_argument("--fusion_dropout", type=float, default=0.1)

    # ================= Loss weights =================
    p.add_argument("--lambda_sa", type=float, default=0.05)  # Structural–Attention
    p.add_argument("--lambda_ds", type=float, default=0.10)  # Dual-Stream
    p.add_argument("--lambda_gr", type=float, default=0.01)  # Gating
    p.add_argument("--lambda_xl", type=float, default=0.10)  # Cross-Modal

    # ================= System =================
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--use_wandb", type=bool, default=False)

    args = p.parse_args()

    # -------- Build SATEConfig --------
    sate_cfg = SATEConfig(
        t5_name=args.model_name,
        max_seq_len=args.max_length,
        max_depth=args.max_depth,
        dropout=args.dropout,
        max_segments=args.max_segments,
    )

    # -------- Build StructuralTransformerConfig --------
    struct_cfg = StructuralTransformerConfig(
        d_model=sate_cfg.d_ast,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.struct_dropout,
        bias_scale_init=args.bias_scale_init,
        bias_channels=args.bias_channels,
    )

    # -------- Paths config --------
    paths = SimpleNamespace(
        train_path=args.train_path,
        val_path=args.val_path,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )

    # -------- Training config --------
    cfg = SimpleNamespace(
        train=bool(args.train),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        fine_tune_t5=args.fine_tune_t5,
        lr=args.lr,
        ema_decay=args.ema_decay,
        log_every=args.log_every,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        model_name=args.model_name,
        device=args.device,
        use_wandb=args.use_wandb,
        sate_cfg=sate_cfg,
        struct_cfg=struct_cfg,
        per_dim_alpha=args.per_dim_alpha,
        fusion_dropout=args.fusion_dropout,
        lambda_sa=args.lambda_sa,
        lambda_ds=args.lambda_ds,
        lambda_gr=args.lambda_gr,
        lambda_xl=args.lambda_xl,
    )

    return paths, cfg


if __name__ == "__main__":
    paths, cfg = parse_args()

    if cfg.use_wandb:
        wandb.login(key="")

    solver = Solver(paths, cfg)
    if cfg.train:
        solver.train()
