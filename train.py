"""
ReiLea Training Module
======================
Trains ReiLeaNet using REINFORCE policy gradient on accumulated PGN games.

Training is triggered automatically after every BATCH_TRIGGER ReiLea-vs-Stockfish
games are played. It runs in a background thread so the server stays responsive.

Algorithm:
  - For each (board_state, move_played, game_outcome) tuple from recent games:
      policy_loss = -log_prob(move) * outcome   # REINFORCE
      value_loss  = MSE(value_pred, outcome)    # value head supervised
  - Outcome: +1 win, -1 loss, 0 draw (from the perspective of the player who moved)
  - After training, updated weights are saved to disk and the live agent reloads them.
"""

import asyncio
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path

from reilea import ReiLeaAgent, encode_board, move_to_index
from chess_game import GameDatabase

BATCH_TRIGGER = 50   # number of ReiLea-vs-SF games before a training run


class ReiLeaTrainer:

    BATCH_TRIGGER = BATCH_TRIGGER   # expose as class attribute for server.py

    def __init__(
        self,
        agent: ReiLeaAgent,
        db: GameDatabase,
        model_path: str,
        lr: float = 1e-4,
        games_per_run: int = 200,
        epochs: int = 3,
        batch_size: int = 512,
    ):
        self.agent = agent
        self.db = db
        self.model_path = model_path
        self.lr = lr
        self.games_per_run = games_per_run
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = optim.Adam(agent.model.parameters(), lr=lr)
        self._optimizer_path = str(Path(model_path).with_suffix(".optimizer.pt"))
        if Path(self._optimizer_path).exists():
            self.optimizer.load_state_dict(torch.load(self._optimizer_path, map_location="cpu"))

        self.game_count = 0          # games accumulated since last training run
        self.is_training = False
        self.last_trained_at: str | None = None
        self.last_result: dict | None = None

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────

    def record_game(self) -> bool:
        """
        Call after each ReiLea-vs-SF game completes.
        Returns True when BATCH_TRIGGER games have accumulated (time to train).
        """
        self.game_count += 1
        return self.game_count >= BATCH_TRIGGER

    async def train_async(self) -> dict:
        """Run training in a thread pool — does not block the FastAPI event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._train)

    @property
    def status(self) -> dict:
        return {
            "is_training": self.is_training,
            "games_since_last_train": self.game_count,
            "games_until_next_train": max(0, BATCH_TRIGGER - self.game_count),
            "batch_trigger": BATCH_TRIGGER,
            "last_trained_at": self.last_trained_at,
            "last_result": self.last_result,
        }

    # ──────────────────────────────────────────────
    #  Internal training logic
    # ──────────────────────────────────────────────

    def _train(self) -> dict:
        """Synchronous training — runs inside a thread pool executor."""
        if self.is_training:
            return {"error": "Already training"}
        self.is_training = True
        try:
            return self._run_training()
        except Exception as e:
            return {"error": str(e)}
        finally:
            self.is_training = False

    def _run_training(self) -> dict:
        games = self._load_recent_reiLea_games()
        if not games:
            return {"error": "No ReiLea games found in database"}

        states, actions, rewards = self._extract_samples(games)
        if not states:
            return {"error": "No valid training samples extracted"}

        device = self.agent.device
        t_states  = torch.tensor(np.array(states),  dtype=torch.float32).to(device)
        t_actions = torch.tensor(actions,            dtype=torch.long).to(device)
        t_rewards = torch.tensor(rewards,            dtype=torch.float32).to(device)

        self.agent.model.train()
        total_ploss = 0.0
        total_vloss = 0.0
        n_batches = 0

        for _ in range(self.epochs):
            perm = torch.randperm(len(t_states), device=device)
            t_states  = t_states[perm]
            t_actions = t_actions[perm]
            t_rewards = t_rewards[perm]

            for i in range(0, len(t_states), self.batch_size):
                bs = t_states [i : i + self.batch_size]
                ba = t_actions[i : i + self.batch_size]
                br = t_rewards[i : i + self.batch_size]

                self.optimizer.zero_grad()
                policy_logits, values = self.agent.model(bs)

                # REINFORCE: reinforce winning moves, discourage losing moves
                log_probs = F.log_softmax(policy_logits, dim=1)
                selected  = log_probs.gather(1, ba.unsqueeze(1)).squeeze(1)
                policy_loss = -(selected * br).mean()

                # Value head: predict game outcome from position
                value_loss = F.mse_loss(values.squeeze(1), br)

                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                n_batches += 1

        self.agent.model.eval()
        self.agent.save(self.model_path)
        torch.save(self.optimizer.state_dict(), self._optimizer_path)

        result = {
            "games_trained_on": len(games),
            "samples": len(states),
            "epochs": self.epochs,
            "avg_policy_loss": round(total_ploss / n_batches, 6) if n_batches else 0,
            "avg_value_loss":  round(total_vloss / n_batches, 6) if n_batches else 0,
            "trained_at": datetime.now().isoformat(),
        }
        self.last_result = result
        self.last_trained_at = result["trained_at"]
        self.game_count = 0   # reset counter after successful training
        return result

    def _load_recent_reiLea_games(self) -> list:
        """Load the most recent `games_per_run` games where ReiLea was a player."""
        all_reiLea = []
        for game in self.db.load_all_games():
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            if "ReiLea" in white or "ReiLea" in black:
                all_reiLea.append(game)
        return all_reiLea[-self.games_per_run:]

    def _extract_samples(self, games: list) -> tuple[list, list, list]:
        """
        Walk each game move-by-move and record (board_encoding, move_idx, reward).
        Reward is from the perspective of the player who made that move:
          +1 if their color won, -1 if lost, 0 for draw.
        """
        states, actions, rewards = [], [], []

        for game in games:
            result = game.headers.get("Result", "*")
            if result == "*":
                continue

            board = game.board()
            for move in game.mainline_moves():
                color = board.turn

                states.append(encode_board(board))
                actions.append(move_to_index(move))

                if result == "1-0":
                    rewards.append(1.0 if color == chess.WHITE else -1.0)
                elif result == "0-1":
                    rewards.append(-1.0 if color == chess.WHITE else 1.0)
                else:
                    rewards.append(0.0)

                board.push(move)

        return states, actions, rewards
