from dataclasses import dataclass, asdict
import argparse
import json
import os
from typing import Optional, List, Dict
import pandas as pd
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from maia2_gnn.dataset import Maia2GNNDataset, collate_graphs
    from maia2_gnn.model import Maia2GNNModel, GNNConfig
    from maia2_gnn.utils import get_all_possible_moves, create_elo_dict, seed_everything
else:
    from .dataset import Maia2GNNDataset, collate_graphs
    from .model import Maia2GNNModel, GNNConfig
    from .utils import get_all_possible_moves, create_elo_dict, seed_everything


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 10
    side_info_coef: float = 1.0
    value_coef: float = 1.0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    test_csv_path: str = ""
    test_max_rows: int = 0
    min_ply: int = 10
    max_ply: int = 160
    max_rows: int = 120000
    patience: int = 5
    save_every: int = 1
    save_dir: str = "./saves"
    show_progress: bool = True
    save_plots: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _make_loader(data, all_moves, all_moves_dict, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    dataset = Maia2GNNDataset(data, all_moves=all_moves, all_moves_dict=all_moves_dict)
    pin_memory = cfg.device.startswith("cuda")
    loader_kwargs = {}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )


def _extract_samples_from_pgn(pgn_path: str, max_rows: int = 4096, min_ply: int = 10, max_ply: int = 160):
    samples = []
    pbar = tqdm(total=max_rows, desc="Parsing PGN", unit="pos", disable=(max_rows <= 0))
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while len(samples) < max_rows:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            white_elo = int(game.headers.get("WhiteElo", "1500") if game.headers.get("WhiteElo", "1500").isdigit() else 1500)
            black_elo = int(game.headers.get("BlackElo", "1500") if game.headers.get("BlackElo", "1500").isdigit() else 1500)
            result = game.headers.get("Result", "1/2-1/2")
            white_win = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)

            board = game.board()
            moves = list(game.mainline_moves())
            for i, move in enumerate(moves):
                fen = board.fen()
                move_uci = move.uci()
                if i >= min_ply and i <= max_ply:
                    if board.turn == chess.WHITE:
                        active_elo, opponent_elo, active_win = white_elo, black_elo, white_win
                    else:
                        active_elo, opponent_elo, active_win = black_elo, white_elo, -white_win
                    samples.append([fen, move_uci, active_elo, opponent_elo, active_win])
                    pbar.update(1)
                board.push(move)
                if len(samples) >= max_rows:
                    break
    pbar.close()
    return samples


def _split_data(data, val_ratio: float, test_ratio: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(data), generator=g).tolist()
    data = [data[i] for i in perm]
    n = len(data)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def _to_active_win(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    if isinstance(value, (int, float)):
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0
    s = str(value).strip().lower()
    if s in {"true", "1", "win", "w"}:
        return 1.0
    if s in {"false", "0", "loss", "l"}:
        return -1.0
    return 0.0


def _load_samples_from_csv(csv_path: str, max_rows: int = 0):
    print(f"Loading CSV test data: {csv_path}")
    df = pd.read_csv(csv_path)
    if "board" not in df.columns or "move" not in df.columns:
        raise ValueError("CSV must contain columns: board, move")
    if "active_elo" not in df.columns or "opponent_elo" not in df.columns:
        raise ValueError("CSV must contain columns: active_elo, opponent_elo")

    if "active_win" in df.columns:
        target_col = "active_win"
    elif "active_won" in df.columns:
        target_col = "active_won"
    else:
        target_col = None

    rows = df[["board", "move", "active_elo", "opponent_elo"]].copy()
    if target_col is None:
        rows["active_win"] = 0.0
    else:
        rows["active_win"] = df[target_col].map(_to_active_win)

    rows = rows.dropna()
    data = rows[["board", "move", "active_elo", "opponent_elo", "active_win"]].values.tolist()
    if max_rows and max_rows > 0:
        data = data[:max_rows]
    print(f"Loaded CSV samples: {len(data)}")
    return data


@torch.no_grad()
def evaluate_split(model, loader, device, ce, bce, mse, cfg: TrainConfig):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        policy_logits, side_logits, value = model(batch)
        y_side = batch.y_side.view(policy_logits.shape[0], -1)
        y_value = batch.y_value.view(-1)
        legal_moves = batch.legal_moves.view(policy_logits.shape[0], -1)
        masked_logits = policy_logits.masked_fill(legal_moves <= 0, -1e9)

        loss_move = ce(policy_logits, batch.y_move)
        loss_side = bce(side_logits, y_side) * cfg.side_info_coef
        loss_value = mse(value, y_value) * cfg.value_coef
        loss = loss_move + loss_side + loss_value
        total_loss += float(loss.item())

        pred = masked_logits.argmax(dim=-1)
        correct += (pred == batch.y_move).sum().item()
        total += batch.y_move.shape[0]
    acc = correct / max(total, 1)
    avg_loss = total_loss / max(len(loader), 1)
    return {"loss": avg_loss, "move_acc": acc, "n": total}


def _save_checkpoint(path, model, optimizer, epoch, metrics, all_moves_len, elo_len, gnn_cfg):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "num_moves": all_moves_len,
            "elo_vocab_size": elo_len,
            "gnn_cfg": asdict(gnn_cfg),
        },
        path,
    )


def run_full_from_pgn(pgn_path: str, cfg: Optional[TrainConfig] = None, gnn_cfg: Optional[GNNConfig] = None) -> Dict[str, object]:
    cfg = cfg or TrainConfig()
    gnn_cfg = gnn_cfg or GNNConfig(hidden_dim=192, num_layers=6, heads=4)
    seed_everything(cfg.seed)

    data = _extract_samples_from_pgn(
        pgn_path,
        max_rows=cfg.max_rows,
        min_ply=cfg.min_ply,
        max_ply=cfg.max_ply,
    )
    if len(data) < 100:
        raise ValueError("Parsed samples too few, please use larger PGN/max_rows.")
    print(f"Parsed PGN samples: {len(data)}")

    train_data, val_data, pgn_test_data = _split_data(data, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    if cfg.test_csv_path:
        test_data = _load_samples_from_csv(cfg.test_csv_path, max_rows=cfg.test_max_rows)
        test_source = "csv"
    else:
        test_data = pgn_test_data
        test_source = "pgn_split"
    print(f"Dataset split -> train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)} ({test_source})")
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    train_loader = _make_loader(train_data, all_moves, all_moves_dict, cfg, shuffle=True)
    val_loader = _make_loader(val_data, all_moves, all_moves_dict, cfg, shuffle=False)
    test_loader = _make_loader(test_data, all_moves, all_moves_dict, cfg, shuffle=False)

    device = torch.device(cfg.device)
    model = Maia2GNNModel(len(all_moves), len(elo_dict), gnn_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    use_amp = cfg.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    os.makedirs(cfg.save_dir, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_loss_move = 0.0
        running_loss_side = 0.0
        running_loss_value = 0.0
        running_correct = 0
        running_total = 0
        train_iter = train_loader
        if cfg.show_progress:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", unit="batch")
        for batch in train_iter:
            batch = batch.to(device, non_blocking=use_amp)
            with torch.cuda.amp.autocast(enabled=use_amp):
                policy_logits, side_logits, value = model(batch)
                y_side = batch.y_side.view(policy_logits.shape[0], -1)
                y_value = batch.y_value.view(-1)
                legal_moves = batch.legal_moves.view(policy_logits.shape[0], -1)

                loss_move = ce(policy_logits, batch.y_move)
                loss_side = bce(side_logits, y_side) * cfg.side_info_coef
                loss_value = mse(value, y_value) * cfg.value_coef
                loss = loss_move + loss_side + loss_value

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_loss_move += float(loss_move.item())
            running_loss_side += float(loss_side.item())
            running_loss_value += float(loss_value.item())
            pred = policy_logits.masked_fill(legal_moves <= 0, -1e9).argmax(dim=-1)
            running_correct += (pred == batch.y_move).sum().item()
            running_total += batch.y_move.shape[0]
            if cfg.show_progress:
                train_iter.set_postfix(
                    loss=f"{running_loss / max(1, running_total / cfg.batch_size):.4f}",
                    acc=f"{running_correct / max(running_total, 1):.4f}",
                )

        train_metrics = {
            "loss": running_loss / max(len(train_loader), 1),
            "loss_move": running_loss_move / max(len(train_loader), 1),
            "loss_side": running_loss_side / max(len(train_loader), 1),
            "loss_value": running_loss_value / max(len(train_loader), 1),
            "move_acc": running_correct / max(running_total, 1),
        }
        val_metrics = evaluate_split(model, val_loader, device, ce, bce, mse, cfg)
        log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_move_acc": train_metrics["move_acc"],
            "val_loss": val_metrics["loss"],
            "val_move_acc": val_metrics["move_acc"],
        }
        history.append(log)
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={log['train_loss']:.4f} "
            f"(move={train_metrics['loss_move']:.4f}, side={train_metrics['loss_side']:.4f}, value={train_metrics['loss_value']:.4f}) "
            f"train_acc={log['train_move_acc']:.4f} "
            f"val_loss={log['val_loss']:.4f} val_acc={log['val_move_acc']:.4f}"
        )

        if epoch % cfg.save_every == 0:
            _save_checkpoint(
                os.path.join(cfg.save_dir, f"epoch_{epoch}.pt"),
                model, optimizer, epoch, log, len(all_moves), len(elo_dict), gnn_cfg,
            )

        if val_metrics["move_acc"] > best_val_acc:
            best_val_acc = val_metrics["move_acc"]
            no_improve = 0
            _save_checkpoint(
                os.path.join(cfg.save_dir, "best.pt"),
                model, optimizer, epoch, log, len(all_moves), len(elo_dict), gnn_cfg,
            )
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience}).")
                break

    _save_checkpoint(
        os.path.join(cfg.save_dir, "last.pt"),
        model, optimizer, epoch, history[-1], len(all_moves), len(elo_dict), gnn_cfg,
    )

    best_ckpt = torch.load(os.path.join(cfg.save_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_split(model, test_loader, device, ce, bce, mse, cfg)
    print(
        f"[test] loss={test_metrics['loss']:.4f} "
        f"move_acc={test_metrics['move_acc']:.4f} n={test_metrics['n']}"
    )

    report = {
        "config": asdict(cfg),
        "gnn_config": asdict(gnn_cfg),
        "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "test_source": test_source,
        "best_val_acc": best_val_acc,
        "test": test_metrics,
        "history": history,
    }
    with open(os.path.join(cfg.save_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    metrics_csv = os.path.join(cfg.save_dir, "metrics.csv")
    pd.DataFrame(history).to_csv(metrics_csv, index=False)
    print(f"Saved metrics csv: {metrics_csv}")

    if cfg.save_plots and len(history) > 0:
        try:
            import matplotlib.pyplot as plt

            epochs = [h["epoch"] for h in history]
            train_loss = [h["train_loss"] for h in history]
            val_loss = [h["val_loss"] for h in history]
            train_acc = [h["train_move_acc"] for h in history]
            val_acc = [h["val_move_acc"] for h in history]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(epochs, train_loss, label="train_loss")
            axes[0].plot(epochs, val_loss, label="val_loss")
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].legend()

            axes[1].plot(epochs, train_acc, label="train_move_acc")
            axes[1].plot(epochs, val_acc, label="val_move_acc")
            axes[1].set_title("Move Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].legend()

            fig.tight_layout()
            plot_path = os.path.join(cfg.save_dir, "training_curves.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved training plot: {plot_path}")
        except Exception as e:
            print(f"Skip plot generation: {e}")
    return report


def run_small_from_pgn(pgn_path: str, save_dir: str = "./saves", max_rows: int = 4096):
    cfg = TrainConfig(
        epochs=1,
        batch_size=64,
        max_rows=max_rows,
        save_dir=save_dir,
        patience=2,
        save_every=1,
    )
    gnn_cfg = GNNConfig(hidden_dim=128, num_layers=3, heads=4)
    report = run_full_from_pgn(pgn_path, cfg=cfg, gnn_cfg=gnn_cfg)
    return os.path.join(save_dir, "best.pt"), report


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Maia2-GNN with full train/val/test pipeline.")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to training PGN file.")
    parser.add_argument("--test_csv_path", type=str, default="", help="Optional final test CSV path (e.g. rapid_sample_data.csv).")
    parser.add_argument("--test_max_rows", type=int, default=0, help="Optional cap for final CSV test rows (0 means all).")
    parser.add_argument("--save_dir", type=str, default="./saves", help="Checkpoint output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs (recommended: 10).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--max_rows", type=int, default=120000, help="Max parsed positions from PGN.")
    parser.add_argument("--patience", type=int, default=5, help="Early stop patience by val move_acc.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--no_plots", action="store_true", help="Disable saving training curves image.")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        max_rows=args.max_rows,
        patience=args.patience,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        device=args.device,
        save_dir=args.save_dir,
        test_csv_path=args.test_csv_path,
        test_max_rows=args.test_max_rows,
        show_progress=(not args.no_progress),
        save_plots=(not args.no_plots),
    )
    gnn_cfg = GNNConfig(hidden_dim=192, num_layers=6, heads=4)
    run_full_from_pgn(args.pgn_path, cfg=cfg, gnn_cfg=gnn_cfg)


if __name__ == "__main__":
    main()
