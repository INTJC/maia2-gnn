import os
import gdown
import pandas as pd
import chess
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from .graph_utils import board_to_graph
from .utils import get_side_info, create_elo_dict, map_to_category


def load_example_test_dataset(save_root: str = "./maia2_data") -> pd.DataFrame:
    url = "https://drive.google.com/uc?id=1fSu4Yp8uYj7xocbHAbjBP6DthsgiJy9X"
    os.makedirs(save_root, exist_ok=True)
    output_path = os.path.join(save_root, "example_test_dataset.csv")
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    data = pd.read_csv(output_path)
    return data[data.move_ply > 10][["board", "move", "active_elo", "opponent_elo"]]


def load_example_train_dataset(save_root: str = "./maia2_data") -> str:
    url = "https://drive.google.com/uc?id=1XBeuhB17z50mFK4tDvPG9rQRbxLSzNqB"
    os.makedirs(save_root, exist_ok=True)
    output_path = os.path.join(save_root, "example_train_dataset.csv")
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    return output_path


class Maia2GNNDataset(Dataset):
    def __init__(self, data, all_moves: list[str], all_moves_dict: dict[str, int]):
        self.data = data
        self.all_moves = all_moves
        self.all_moves_dict = all_moves_dict
        self.elo_dict = create_elo_dict()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_fen, move_uci, elo_self, elo_oppo, active_win = self.data[idx]
        board = chess.Board(board_fen)
        legal_moves, side_info = get_side_info(board, move_uci, self.all_moves_dict)
        elo_self = map_to_category(int(elo_self), self.elo_dict)
        elo_oppo = map_to_category(int(elo_oppo), self.elo_dict)
        graph = board_to_graph(
            board=board,
            all_moves=self.all_moves,
            all_moves_dict=self.all_moves_dict,
            move_uci=move_uci,
            side_info=side_info,
            value=float(active_win),
            elo_self=elo_self,
            elo_oppo=elo_oppo,
            legal_moves_vec=legal_moves,
        )
        return graph


def collate_graphs(items):
    return Batch.from_data_list(items)


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    for batch in dataloader:
        batch = batch.to(device)
        policy_logits, _, _ = model(batch)
        legal_moves = batch.legal_moves.view(policy_logits.shape[0], -1)
        masked_logits = policy_logits.masked_fill(legal_moves <= 0, -1e9)
        pred = masked_logits.argmax(dim=-1)
        correct += (pred == batch.y_move).sum().item()
        total += batch.y_move.shape[0]
    return correct, total
