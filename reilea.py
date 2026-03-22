import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for i, pt in enumerate(PIECE_TYPES):
        for sq in board.pieces(pt, chess.WHITE):
            planes[i, sq // 8, sq % 8] = 1.0
        for sq in board.pieces(pt, chess.BLACK):
            planes[i + 6, sq // 8, sq % 8] = 1.0
    planes[12] = float(board.turn == chess.WHITE)
    planes[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        planes[17, board.ep_square // 8, board.ep_square % 8] = 1.0
    return planes


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    from_sq = idx // 64
    to_sq = idx % 64
    move = chess.Move(from_sq, to_sq)
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
        move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    return move


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ReiLeaNet(nn.Module):
    def __init__(self, num_res_blocks: int = 6, channels: int = 128):
        super().__init__()
        self.conv_in = nn.Conv2d(18, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)

        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value


class ReiLeaAgent:
    def __init__(self, model_path: str = None, device: str = None, temperature: float = 0.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.model = ReiLeaNet().to(self.device)
        self.model.eval()
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)

    def predict_move(self, board: chess.Board) -> str:
        planes = encode_board(board)
        x = torch.tensor(planes).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(x)

        legal_indices = [move_to_index(m) for m in board.legal_moves]
        if not legal_indices:
            return None

        logits = policy_logits[0].cpu().numpy()
        legal_logits = np.array([logits[i] for i in legal_indices])

        if self.temperature > 0:
            legal_logits = legal_logits / self.temperature
            probs = np.exp(legal_logits - legal_logits.max())
            probs /= probs.sum()
            chosen_idx = np.random.choice(len(legal_indices), p=probs)
        else:
            chosen_idx = int(np.argmax(legal_logits))

        move = index_to_move(legal_indices[chosen_idx], board)
        return move.uci()

    def predict_with_log_prob(self, board: chess.Board):
        planes = encode_board(board)
        x = torch.tensor(planes).unsqueeze(0).to(self.device)

        policy_logits, value = self.model(x)

        legal_moves = list(board.legal_moves)
        legal_indices = [move_to_index(m) for m in legal_moves]

        legal_logits = policy_logits[0][legal_indices]
        log_probs = F.log_softmax(legal_logits, dim=0)
        probs = torch.exp(log_probs)

        chosen_idx = torch.multinomial(probs, 1).item()
        move = index_to_move(legal_indices[chosen_idx], board)

        return move.uci(), log_probs[chosen_idx], value[0, 0]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
