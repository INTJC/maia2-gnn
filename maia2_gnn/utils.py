import chess
import random
import numpy as np
import torch
import yaml
import os
import pickle
import time
import pyzstd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def parse_args(cfg_file_path: str):
    with open(cfg_file_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return Config(cfg_dict)


def create_elo_dict() -> dict[str, int]:
    interval = 100
    start = 1100
    end = 2000
    range_dict: dict[str, int] = {f"<{start}": 0}
    idx = 1
    for low in range(start, end, interval):
        high = low + interval
        range_dict[f"{low}-{high - 1}"] = idx
        idx += 1
    range_dict[f">={end}"] = idx
    return range_dict


def map_to_category(elo: int, elo_dict: dict[str, int]) -> int:
    interval = 100
    start = 1100
    end = 2000
    if elo < start:
        return elo_dict[f"<{start}"]
    if elo >= end:
        return elo_dict[f">={end}"]
    for low in range(start, end, interval):
        high = low + interval
        if low <= elo < high:
            return elo_dict[f"{low}-{high - 1}"]
    raise ValueError(f"Invalid elo: {elo}")


def get_side_info(board: chess.Board, move_uci: str, all_moves_dict: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    move = chess.Move.from_uci(move_uci)
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    if moving_piece is None:
        raise ValueError(f"Invalid move for board: {move_uci}")

    from_square_encoded = torch.zeros(64, dtype=torch.float32)
    from_square_encoded[move.from_square] = 1.0
    to_square_encoded = torch.zeros(64, dtype=torch.float32)
    to_square_encoded[move.to_square] = 1.0

    if move_uci == "e1g1":
        rook_move = chess.Move.from_uci("h1f1")
        from_square_encoded[rook_move.from_square] = 1.0
        to_square_encoded[rook_move.to_square] = 1.0
    elif move_uci == "e1c1":
        rook_move = chess.Move.from_uci("a1d1")
        from_square_encoded[rook_move.from_square] = 1.0
        to_square_encoded[rook_move.to_square] = 1.0

    board.push(move)
    is_check = board.is_check()
    board.pop()

    side_info = torch.zeros(6 + 6 + 1, dtype=torch.float32)
    side_info[moving_piece.piece_type - 1] = 1.0
    if move_uci in {"e1g1", "e1c1"}:
        side_info[3] = 1.0
    if captured_piece is not None:
        side_info[6 + captured_piece.piece_type - 1] = 1.0
    if is_check:
        side_info[-1] = 1.0

    legal_moves = torch.zeros(len(all_moves_dict), dtype=torch.float32)
    legal_idx = torch.tensor([all_moves_dict[m.uci()] for m in board.legal_moves], dtype=torch.long)
    legal_moves[legal_idx] = 1.0
    side_info = torch.cat([side_info, from_square_encoded, to_square_encoded, legal_moves], dim=0)
    return legal_moves, side_info


def generate_pawn_promotions() -> list[str]:
    promotion_rows = {"white": "7", "black": "2"}
    promotion_pieces = ["q", "r", "b", "n"]
    promotions: list[str] = []
    for color, row in promotion_rows.items():
        target_row = "8" if color == "white" else "1"
        for file in "abcdefgh":
            for piece in promotion_pieces:
                promotions.append(f"{file}{row}{file}{target_row}{piece}")
            if file != "a":
                left_file = chr(ord(file) - 1)
                for piece in promotion_pieces:
                    promotions.append(f"{file}{row}{left_file}{target_row}{piece}")
            if file != "h":
                right_file = chr(ord(file) + 1)
                for piece in promotion_pieces:
                    promotions.append(f"{file}{row}{right_file}{target_row}{piece}")
    return promotions


def get_all_possible_moves() -> list[str]:
    all_moves: list[str] = []
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            all_moves.extend(list(board.legal_moves))
            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            all_moves.extend(list(board.legal_moves))
    all_moves_uci = [m.uci() for m in all_moves]
    return all_moves_uci + generate_pawn_promotions()


def get_chunks(pgn_path: str, chunk_size: int):
    chunks = []
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        while True:
            start_pos = pgn_file.tell()
            game_count = 0
            while game_count < chunk_size:
                line = pgn_file.readline()
                if not line:
                    break
                if line.endswith("1-0\n") or line.endswith("0-1\n") or line.endswith("1/2-1/2\n") or line.endswith("*\n"):
                    game_count += 1
            line = pgn_file.readline()
            if line not in ["\n", ""]:
                raise ValueError("Invalid PGN chunk boundary")
            end_pos = pgn_file.tell()
            chunks.append((start_pos, end_pos))
            if not line:
                break
    return chunks


def read_or_create_chunks(pgn_path: str, chunk_size: int):
    cache_file = pgn_path.replace(".pgn", "_chunks.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    start = time.time()
    pgn_chunks = get_chunks(pgn_path, chunk_size)
    print(f"Chunking cost: {time.time() - start:.2f}s")
    with open(cache_file, "wb") as f:
        pickle.dump(pgn_chunks, f)
    return pgn_chunks


def decompress_zst(file_path: str, decompressed_path: str):
    with open(file_path, "rb") as compressed_file, open(decompressed_path, "wb") as decompressed_file:
        pyzstd.decompress_stream(compressed_file, decompressed_file)
