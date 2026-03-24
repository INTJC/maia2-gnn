import chess
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from .graph_utils import board_to_graph
from .utils import get_all_possible_moves, create_elo_dict, map_to_category


def _mirror_square(square: str) -> str:
    return square[0] + str(9 - int(square[1]))


def mirror_move(move_uci: str) -> str:
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion = move_uci[4:] if len(move_uci) > 4 else ""
    return _mirror_square(start_square) + _mirror_square(end_square) + promotion


def preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves, all_moves_dict):
    if fen.split(" ")[1] == "w":
        board = chess.Board(fen)
        black_flag = False
    elif fen.split(" ")[1] == "b":
        board = chess.Board(fen).mirror()
        black_flag = True
    else:
        raise ValueError(f"Invalid fen: {fen}")

    elo_self = map_to_category(int(elo_self), elo_dict)
    elo_oppo = map_to_category(int(elo_oppo), elo_dict)
    legal_moves = torch.zeros(len(all_moves_dict), dtype=torch.float32)
    legal_idx = torch.tensor([all_moves_dict[m.uci()] for m in board.legal_moves], dtype=torch.long)
    legal_moves[legal_idx] = 1.0
    graph = board_to_graph(
        board=board,
        all_moves=all_moves,
        all_moves_dict=all_moves_dict,
        elo_self=elo_self,
        elo_oppo=elo_oppo,
        legal_moves_vec=legal_moves,
    )
    return graph, black_flag


class TestDataset(Dataset):
    def __init__(self, data: pd.DataFrame, all_moves, all_moves_dict, elo_dict):
        self.data = data.values.tolist()
        self.all_moves = all_moves
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, _, elo_self, elo_oppo = self.data[idx]
        graph, black_flag = preprocessing(
            fen, elo_self, elo_oppo, self.elo_dict, self.all_moves, self.all_moves_dict
        )
        graph.black_flag = torch.tensor(int(black_flag), dtype=torch.long)
        return fen, graph


def _collate(items):
    fens = [x[0] for x in items]
    graphs = Batch.from_data_list([x[1] for x in items])
    return fens, graphs


def get_preds(model, dataloader, all_moves_dict_reversed):
    move_probs, win_probs = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for fens, batch in dataloader:
            batch = batch.to(device)
            logits_maia, _, logits_value = model(batch)
            legal_moves = batch.legal_moves.view(logits_maia.shape[0], -1)
            logits_maia_legal = logits_maia.masked_fill(legal_moves <= 0, -1e9)
            probs = logits_maia_legal.softmax(dim=-1).cpu()
            values = (logits_value / 2 + 0.5).clamp(0, 1).cpu()

            for i, fen in enumerate(fens):
                black_flag = fen.split(" ")[1] == "b"
                v = values[i].item()
                if black_flag:
                    v = 1 - v
                win_probs.append(round(v, 4))

                legal_idx = (legal_moves[i] > 0).nonzero().flatten().cpu().tolist()
                each = {}
                for idx in legal_idx:
                    mv = all_moves_dict_reversed[idx]
                    if black_flag:
                        mv = mirror_move(mv)
                    each[mv] = round(float(probs[i, idx]), 4)
                move_probs.append(dict(sorted(each.items(), key=lambda item: item[1], reverse=True)))
    return move_probs, win_probs


def inference_batch(data, model, verbose=1, batch_size=512, num_workers=0):
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    dataset = TestDataset(data, all_moves, all_moves_dict, elo_dict)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=_collate
    )
    move_probs, win_probs = get_preds(model, dataloader, all_moves_dict_reversed)
    data = data.copy()
    data["win_probs"] = win_probs
    data["move_probs"] = move_probs
    acc = 0
    for i in range(len(data)):
        best = max(move_probs[i], key=move_probs[i].get)
        if best == data.iloc[i]["move"]:
            acc += 1
    return data, round(acc / len(data), 4)


def prepare():
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    return [all_moves, all_moves_dict, elo_dict, all_moves_dict_reversed]


def inference_each(model, prepared, fen, elo_self, elo_oppo):
    all_moves, all_moves_dict, elo_dict, all_moves_dict_reversed = prepared
    graph, black_flag = preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves, all_moves_dict)
    device = next(model.parameters()).device
    batch = Batch.from_data_list([graph]).to(device)
    model.eval()
    with torch.no_grad():
        logits_maia, _, logits_value = model(batch)
        legal_moves = batch.legal_moves.view(logits_maia.shape[0], -1)
        logits_maia_legal = logits_maia.masked_fill(legal_moves <= 0, -1e9)
        probs = logits_maia_legal.softmax(dim=-1).cpu()
        value = (logits_value / 2 + 0.5).clamp(0, 1).item()
        if black_flag:
            value = 1 - value
        legal_idx = (legal_moves[0] > 0).nonzero().flatten().cpu().tolist()
        move_probs = {}
        for idx in legal_idx:
            mv = all_moves_dict_reversed[idx]
            if black_flag:
                mv = mirror_move(mv)
            move_probs[mv] = round(float(probs[0, idx]), 4)
        move_probs = dict(sorted(move_probs.items(), key=lambda item: item[1], reverse=True))
    return move_probs, round(value, 4)
