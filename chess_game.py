import chess
import chess.pgn
import chess.engine
import datetime
import json
import random
import shutil
from pathlib import Path

STOCKFISH_PATH = shutil.which("stockfish") or "stockfish"


class GameDatabase:

    def __init__(self, db_dir: str = "game_data"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.pgn_path = self.db_dir / "games.pgn"
        self.index_path = self.db_dir / "index.json"
        self._load_index()

    def _load_index(self):
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {"games": [], "total_games": 0}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def save_game(self, game: chess.pgn.Game) -> int:
        game_id = self.index["total_games"] + 1

        game.headers["GameID"] = str(game_id)
        if "Date" not in game.headers or game.headers["Date"] == "????.??.??":
            game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")

        with open(self.pgn_path, "a") as f:
            f.write(str(game))
            f.write("\n\n")

        moves = list(game.mainline_moves())
        entry = {
            "game_id": game_id,
            "white": game.headers.get("White", "Unknown"),
            "black": game.headers.get("Black", "Unknown"),
            "result": game.headers.get("Result", "*"),
            "date": game.headers.get("Date", ""),
            "eco": game.headers.get("ECO", ""),
            "num_moves": len(moves),
            "termination": game.headers.get("Termination", ""),
        }
        self.index["games"].append(entry)
        self.index["total_games"] = game_id
        self._save_index()

        return game_id

    def load_all_games(self):
        if not self.pgn_path.exists():
            return
        with open(self.pgn_path, "r") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game

    def import_pgn(self, pgn_file: str) -> int:
        count = 0
        with open(pgn_file, "r") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                self.save_game(game)
                count += 1
        return count

    def get_stats(self) -> dict:
        if not self.index["games"]:
            return {"total_games": 0}

        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        total_moves = 0
        for g in self.index["games"]:
            results[g.get("result", "*")] = results.get(g.get("result", "*"), 0) + 1
            total_moves += g.get("num_moves", 0)

        return {
            "total_games": self.index["total_games"],
            "results": results,
            "avg_moves": total_moves / len(self.index["games"]) if self.index["games"] else 0,
        }


class ChessGame:

    def __init__(
        self,
        white: str = "Human",
        black: str = "Human",
        event: str = "Local Game",
        sf_skill: int = 10,
    ):
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = event
        self.game.headers["Site"] = "Local"
        self.game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
        self.game.headers["White"] = white
        self.game.headers["Black"] = black
        self.game.headers["Result"] = "*"
        self.node = self.game
        self.move_history = []
        self._sf_engine = None
        self._sf_skill = sf_skill

    def _init_stockfish(self):
        self._sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self._sf_engine.configure({"Skill Level": self._sf_skill})

    def close(self):
        if self._sf_engine:
            self._sf_engine.quit()
            self._sf_engine = None

    def get_legal_moves(self) -> list[str]:
        return [
            {"uci": m.uci(), "san": self.board.san(m)}
            for m in self.board.legal_moves
        ]

    def make_move(self, move_str: str) -> dict:
        move = None

        try:
            move = chess.Move.from_uci(move_str)
            if move not in self.board.legal_moves:
                move = None
        except ValueError:
            pass

        if move is None:
            try:
                move = self.board.parse_san(move_str)
            except (chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
                return {"error": str(e), "legal_moves": self.get_legal_moves()}

        if move is None:
            return {"error": f"Invalid move: {move_str}", "legal_moves": self.get_legal_moves()}

        san = self.board.san(move)
        self.node = self.node.add_variation(move)
        self.board.push(move)
        self.move_history.append(san)

        result = {
            "move": san,
            "uci": move.uci(),
            "fen": self.board.fen(),
            "move_number": self.board.fullmove_number,
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over(),
        }

        if self.board.is_game_over():
            result["result"] = self.board.result()
            self.game.headers["Result"] = self.board.result()
            if self.board.is_checkmate():
                self.game.headers["Termination"] = "checkmate"
            elif self.board.is_stalemate():
                self.game.headers["Termination"] = "stalemate"
            else:
                self.game.headers["Termination"] = "draw"

        return result

    def make_random_move(self) -> dict:
        legal = list(self.board.legal_moves)
        if not legal:
            return {"error": "No legal moves"}
        return self.make_move(random.choice(legal).uci())

    def make_stockfish_move(self, time_limit: float = 0.1) -> dict:
        if self._sf_engine is None:
            self._init_stockfish()
        result = self._sf_engine.play(self.board, chess.engine.Limit(time=time_limit))
        return self.make_move(result.move.uci())

    def make_reiLea_move(self, agent) -> dict:
        uci = agent.predict_move(self.board)
        if uci is None:
            return self.make_random_move()
        return self.make_move(uci)

    def resign(self):
        if self.board.turn == chess.WHITE:
            self.game.headers["Result"] = "0-1"
            self.game.headers["Termination"] = "resignation"
        else:
            self.game.headers["Result"] = "1-0"
            self.game.headers["Termination"] = "resignation"
        return self.game.headers["Result"]

    def get_pgn(self) -> str:
        return str(self.game)

    def get_board_ascii(self) -> str:
        return str(self.board)

    def get_fen(self) -> str:
        return self.board.fen()
