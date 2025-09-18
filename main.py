import argparse
from types import SimpleNamespace

import wandb
from solver import Solver


def parse_args():
    p = argparse.ArgumentParser()

    # Data / IO
    p.add_argument("--train_path", default="./data/train_dataset.jsonl")
    p.add_argument("--val_path", default="./data/val_dataset.jsonl")
    p.add_argument("--data_dir", default="saved_data")
    p.add_argument("--model_dir", default="saved_models")

    # Training
    p.add_argument("--train", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--log_every", type=int, default=20)

    # model
    p.add_argument("--model_name", type=str, default="t5-base")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--is_load_state_dict", type=bool, default=True)

    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])

    args = p.parse_args()

    # Tách cấu hình thành SimpleNamespace để dễ truy cập
    paths = SimpleNamespace(
        train_path=args.train_path,
        val_path=args.val_path,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )

    cfg = SimpleNamespace(
        train=bool(args.train),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        ema_decay=args.ema_decay,
        log_every=args.log_every,
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        is_load_state_dict=args.is_load_state_dict,
    )

    return paths, cfg


if __name__ == "__main__":
    paths, cfg = parse_args()

    # Nếu dùng wandb
    wandb.login(key="d83175b72ab7d073e2ed4f0e60ef001c11cd4555")

    # Giả sử Solver là class huấn luyện
    solver = Solver(paths, cfg)
    solver.train()
