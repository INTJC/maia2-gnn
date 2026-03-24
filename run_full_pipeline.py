import argparse
import os
import sys
import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maia2_gnn.model import GNNConfig
from maia2_gnn.train import TrainConfig, run_full_from_pgn


def parse_args():
    parser = argparse.ArgumentParser(description="One-click full train/val/test pipeline for maia2-gnn.")
    parser.add_argument(
        "--pgn_path",
        type=str,
        default="maia2_gnn/example_training_data.pgn",
        help="PGN for train/val split.",
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        default="maia2_gnn/rapid_sample_data.csv",
        help="CSV for final test evaluation.",
    )
    parser.add_argument("--save_dir", type=str, default="./saves/full_run", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--max_rows", type=int, default=120000)
    parser.add_argument("--test_max_rows", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def main():
    args = parse_args()
    pgn_path = resolve_path(args.pgn_path)
    test_csv_path = resolve_path(args.test_csv_path) if args.test_csv_path else ""
    save_dir = resolve_path(args.save_dir)

    if not os.path.exists(pgn_path):
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")
    if test_csv_path and not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV file not found: {test_csv_path}")

    selected_device = args.device
    if selected_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        selected_device = "cpu"

    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_rows=args.max_rows,
        patience=args.patience,
        device=selected_device,
        save_dir=save_dir,
        test_csv_path=test_csv_path,
        test_max_rows=args.test_max_rows,
        save_plots=(not args.no_plots),
    )
    gnn_cfg = GNNConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    )

    print("Running full pipeline with:")
    print(f"  pgn_path={pgn_path}")
    print(f"  test_csv_path={test_csv_path or 'None'}")
    print(f"  save_dir={save_dir}")
    print(f"  device={selected_device}")
    report = run_full_from_pgn(pgn_path=pgn_path, cfg=cfg, gnn_cfg=gnn_cfg)
    print("Done.")
    print(f"Best val acc: {report['best_val_acc']:.4f}")
    print(f"Test move acc: {report['test']['move_acc']:.4f}")
    print(f"Report: {os.path.join(save_dir, 'train_report.json')}")
    print(f"Metrics: {os.path.join(save_dir, 'metrics.csv')}")
    if not args.no_plots:
        print(f"Plot: {os.path.join(save_dir, 'training_curves.png')}")


if __name__ == "__main__":
    main()
