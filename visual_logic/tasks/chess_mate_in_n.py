import io
import json
import random
from pathlib import Path
from typing import List, Optional, TypedDict

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image

from visual_logic.tasks.base import Task, TaskProblem


class ChessMateSpecificMetadata(TypedDict, total=False):
    mate_in: int
    initial_turn: str  # "w" or "b"
    moves_san: List[str]


WHITE = True
BLACK = False

PIECE_ENCODING = {
    (chess.PAWN, WHITE): 1,
    (chess.KNIGHT, WHITE): 2,
    (chess.BISHOP, WHITE): 3,
    (chess.ROOK, WHITE): 4,
    (chess.QUEEN, WHITE): 5,
    (chess.KING, WHITE): 6,
    (chess.PAWN, BLACK): 7,
    (chess.KNIGHT, BLACK): 8,
    (chess.BISHOP, BLACK): 9,
    (chess.ROOK, BLACK): 10,
    (chess.QUEEN, BLACK): 11,
    (chess.KING, BLACK): 12,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Convert chess.Board into fixed numeric 8x8 grid."""
    grid = np.zeros((8, 8), dtype=int)
    for square, piece in board.piece_map().items():
        r, c = divmod(square, 8)
        grid[7 - r, c] = PIECE_ENCODING[(piece.piece_type, piece.color)]
    return grid


###
# Utils to visualize
###

from typing import List


def decode_grid_to_board(grid: List[List[int]]) -> chess.Board:
    board = chess.Board.empty()
    inv_map = {v: k for k, v in PIECE_ENCODING.items()}
    for r in range(8):
        for c in range(8):
            val = grid[r][c]
            if val != 0:
                piece_type, color = inv_map[val]
                square = chess.square(c, 7 - r)  # reverse of encode_board
                board.set_piece_at(square, chess.Piece(piece_type, color))
    return board


def render_as_chess_map(grid: List[List[int]], to_file: str = "board.svg"):
    board = decode_grid_to_board(grid)
    svg_code = chess.svg.board(board=board)
    with open(to_file, "w") as f:
        f.write(svg_code)
    return to_file


def render_as_chess_maps(grids: List[List[List[int]]], to_file: str = "boards.png"):
    images = []
    for grid in grids:
        board = decode_grid_to_board(grid)
        svg_code = chess.svg.board(board=board)
        png_bytes = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
        img = Image.open(io.BytesIO(png_bytes))
        images.append(img)

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    new_img.save(to_file)
    return to_file


####


class ChessMate(Task):
    def __init__(
        self,
        mate_in: int = 1,
        initial_turn: str = "w",
        seed: Optional[int] = None,
        problem_id: Optional[int] = None,
    ):
        self.mate_in = mate_in
        self.initial_turn = initial_turn
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.problem_id = problem_id

        chess_games_json = (
            Path(__file__).resolve().parent / "resources" / "chess_mate_in_123.json"
        )
        with open(chess_games_json, "r") as f:
            self.data = json.load(f)

        # Filter dataset
        self.filtered = [
            g
            for g in self.data
            if f"mateIn{self.mate_in}" in g["Themes"].split()
            and g["FEN"].split()[1] == self.initial_turn
        ]

    def generate(self) -> TaskProblem:
        if not self.filtered:
            raise ValueError(
                f"No positions found with mate in {self.mate_in} for {self.initial_turn}"
            )
        if self.problem_id is None:
            game_data = random.choice(self.filtered)
        else:
            game_data = self.filtered[self.problem_id]

        fen = game_data["FEN"]
        moves_san = game_data["MovesSAN"].split()

        board = chess.Board(fen)
        init_grid = encode_board(board)

        inter_grids = []
        for i, move_san in enumerate(moves_san):
            move = board.parse_san(move_san)
            board.push(move)
            if i < len(moves_san) - 1:
                inter_grids.append(encode_board(board))

        tgt_grid = encode_board(board)

        metadata: ChessMateSpecificMetadata = {
            "mate_in": self.mate_in,
            "initial_turn": self.initial_turn,
            "moves_san": moves_san,
        }

        return TaskProblem(
            init_grid=init_grid.tolist(),
            tgt_grid=tgt_grid.tolist(),
            intermediate_grids=(
                [g.tolist() for g in inter_grids] if inter_grids else None
            ),
            task_specific_metadata=metadata,
        )
