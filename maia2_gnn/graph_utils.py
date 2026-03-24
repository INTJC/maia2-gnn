from dataclasses import dataclass
from typing import Optional
import chess
import torch
from torch_geometric.data import Data


PROMOTION_TO_IDX = {"q": 0, "r": 1, "b": 2, "n": 3}


@dataclass(frozen=True)
class GraphFeatureSpec:
    node_dim: int = 20
    edge_dim: int = 20


def _square_coord_features(square: int) -> tuple[float, float]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return file_idx / 7.0, rank_idx / 7.0


def _node_features(board: chess.Board) -> torch.Tensor:
    x = torch.zeros((64, GraphFeatureSpec.node_dim), dtype=torch.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            base = 0 if piece.color == chess.WHITE else 6
            x[sq, base + piece.piece_type - 1] = 1.0
        x[sq, 12] = 1.0 if board.turn == chess.WHITE else 0.0
        x[sq, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        x[sq, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        x[sq, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        x[sq, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        x[sq, 17] = 1.0 if board.ep_square == sq else 0.0
        file_f, rank_f = _square_coord_features(sq)
        x[sq, 18] = file_f
        x[sq, 19] = rank_f
    return x


def _edge_features(board: chess.Board, from_sq: int, to_sq: int, promo: str, is_legal: bool) -> torch.Tensor:
    edge = torch.zeros((GraphFeatureSpec.edge_dim,), dtype=torch.float32)
    edge[0] = 1.0 if is_legal else 0.0
    edge[1] = float(chess.square_file(to_sq) - chess.square_file(from_sq))
    edge[2] = float(chess.square_rank(to_sq) - chess.square_rank(from_sq))
    if promo in PROMOTION_TO_IDX:
        edge[3 + PROMOTION_TO_IDX[promo]] = 1.0

    moving_piece = board.piece_at(from_sq)
    if moving_piece is not None:
        base = 7 if moving_piece.color == chess.WHITE else 13
        edge[base + moving_piece.piece_type - 1] = 1.0
    return edge


def board_to_graph(
    board: chess.Board,
    all_moves: list[str],
    all_moves_dict: dict[str, int],
    move_uci: Optional[str] = None,
    side_info: Optional[torch.Tensor] = None,
    value: Optional[float] = None,
    elo_self: Optional[int] = None,
    elo_oppo: Optional[int] = None,
    legal_moves_vec: Optional[torch.Tensor] = None,
) -> Data:
    x = _node_features(board)
    legal_set = {m.uci() for m in board.legal_moves}

    senders = []
    receivers = []
    edge_attr = []
    for uci in all_moves:
        mv = chess.Move.from_uci(uci)
        senders.append(mv.from_square)
        receivers.append(mv.to_square)
        promo = uci[4:] if len(uci) > 4 else ""
        edge_attr.append(_edge_features(board, mv.from_square, mv.to_square, promo, uci in legal_set))

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_attr_t = torch.stack(edge_attr, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t)
    if move_uci is not None:
        data.y_move = torch.tensor(all_moves_dict[move_uci], dtype=torch.long)
    if side_info is not None:
        data.y_side = side_info.float()
    if value is not None:
        data.y_value = torch.tensor(float(value), dtype=torch.float32)
    if elo_self is not None:
        data.elo_self = torch.tensor(int(elo_self), dtype=torch.long)
    if elo_oppo is not None:
        data.elo_oppo = torch.tensor(int(elo_oppo), dtype=torch.long)
    if legal_moves_vec is not None:
        data.legal_moves = legal_moves_vec.float()
    return data
