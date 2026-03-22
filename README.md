# Chess RL

A full-stack chess application with a browser-based board, a REST API backend, and a custom reinforcement-learning chess engine called ReiLea. The project is split into three layers that are designed to work together: a game engine and database, a web server, and a neural-network agent. A separate training directory contains the Jupyter notebook and data utilities needed to train the agent from scratch.

---

## Project Structure

```
chess/
├── chess_game.py              # Core game logic, move validation, PGN database
├── server.py                  # FastAPI web server and REST API
├── reilea.py                  # ReiLea neural network architecture and agent
├── train.py                   # Live RL training loop (REINFORCE, auto-triggered)
├── static/
│   └── index.html             # Single-page browser frontend
├── RL Training/
│   ├── rl_utils.py            # Standalone data extraction utilities
│   └── RL engine.ipynb        # Two-phase training notebook (SL then RL)
├── game_data/                 # Created at runtime
│   ├── games.pgn              # All saved games in PGN format
│   └── index.json             # JSON metadata index for fast lookups
└── models/                    # Created at runtime
    ├── reiLea_supervised.pt   # Model checkpoint after supervised training
    └── reiLea_rl.pt           # Model checkpoint after RL fine-tuning
```

---

## File Descriptions

### chess_game.py

This is the foundation of the project. It defines two classes.

`GameDatabase` manages persistent storage of chess games. It keeps games in an append-only PGN file (`game_data/games.pgn`) and maintains a parallel JSON index (`game_data/index.json`) with lightweight metadata such as player names, result, date, number of moves, and ECO code. The index allows the server to query statistics and paginate game listings without re-parsing the full PGN file. `GameDatabase` also supports bulk import from external PGN files, which is the mechanism for loading Lichess or chess.com datasets.

`ChessGame` represents a single active game. It wraps a `python-chess` board, a PGN game tree, and move history. It exposes methods for:
- Validating and applying moves (accepting both UCI and SAN notation)
- Querying legal moves, FEN, board ASCII, and check/checkmate/stalemate state
- Making moves on behalf of three types of agents: random selection, Stockfish (via UCI protocol), and the ReiLea neural network agent
- Resigning
- Exporting the current game as a PGN string

When a game ends, the caller (typically `server.py`) is responsible for passing the finished game object to `GameDatabase.save_game()`.

This file also exports `STOCKFISH_PATH`, which resolves the Stockfish binary via `shutil.which` and falls back to the string `"stockfish"`. Other files import this constant directly.

---

### reilea.py

This file defines the ReiLea engine, which is both the neural network architecture and the inference agent used during gameplay.

`encode_board(board)` converts a `chess.Board` into an 18-channel 8x8 float32 tensor. Channels 0-5 encode white piece occupancy (pawn through king), channels 6-11 encode black piece occupancy, channel 12 encodes whose turn it is, channels 13-16 encode castling rights for both sides, and channel 17 encodes the en passant square. This 18-plane representation is richer than a basic 12-plane encoding and is what `ReiLeaNet` was designed to consume.

`move_to_index(move)` and `index_to_move(idx, board)` convert between `chess.Move` objects and a flat integer in [0, 4095] using the formula `from_square * 64 + to_square`. Pawn promotions are automatically resolved to queen promotions in `index_to_move`. This 4096-class action space is the output dimension of the policy head.

`ResBlock` is a standard residual block with two 3x3 convolutions, batch normalization, and a ReLU skip connection.

`ReiLeaNet` is the main neural network. Its forward pass is:
1. An input convolution projecting 18 input channels to 128 channels
2. Six residual blocks
3. A policy head: 1x1 convolution to 2 channels, flatten, fully-connected to 4096 logits (one per possible from-square/to-square pair)
4. A value head: 1x1 convolution to 1 channel, flatten, two fully-connected layers ending with tanh to produce a scalar in [-1, 1]

`ReiLeaAgent` wraps `ReiLeaNet` for inference. It loads weights from disk at construction time (if the path exists and the file is present; otherwise it runs with random weights). `predict_move(board)` runs a forward pass, masks the 4096 logits down to only legal moves, and returns the best move as a UCI string. If `temperature > 0` it samples from a softmax distribution instead of taking the argmax. `predict_with_log_prob(board)` is used during RL training and returns the chosen move, its log-probability, and the value prediction so the training loop can compute policy gradient updates.

---

### server.py

This is the FastAPI application that connects all components. At startup it instantiates a single `GameDatabase`, loads a single `ReiLeaAgent` from `models/reiLea_rl.pt`, and defines two in-memory dictionaries: `active_games` maps session IDs to `ChessGame` instances, and `game_modes` maps session IDs to mode strings.

Game modes determine how the bot endpoints behave. The modes are: `pvp` (human vs human), `bot-white` and `bot-black` (Stockfish plays the designated side), `reiLea-white` and `reiLea-black` (ReiLea plays the designated side), `sf-vs-sf` (both sides Stockfish), and `reiLea-vs-sf` (ReiLea plays white, Stockfish plays black). The helper `_bot_move_for_game(game, mode)` reads the current board turn and the mode to decide which move method to call.

The REST endpoints are:

- `POST /api/game/new` — creates a new `ChessGame` with a random 8-character UUID, stores it in `active_games`, and returns the initial FEN and legal moves.
- `POST /api/game/{game_id}/move` — applies a human move to the game. If the game ends it saves it to the database, cleans up the active game, and returns the saved game ID.
- `POST /api/game/{game_id}/bot-move` — triggers a bot move according to the game's mode. Same save-and-cleanup logic applies on game over.
- `POST /api/game/{game_id}/resign` — records a resignation result, saves the game, and cleans up.
- `GET /api/game/{game_id}/state` — returns the full current state including FEN, legal moves, move history, PGN, check status, and mode.
- `GET /api/game/{game_id}/pgn` — returns the raw PGN text.
- `GET /api/db/stats` — returns total game count and result breakdown from the index.
- `GET /api/db/games` — paginates the game metadata index with `limit` and `offset` query parameters.
- `POST /api/db/generate` — generates `n` ReiLea self-play games (ReiLea vs ReiLea) and saves them.
- `POST /api/db/generate-sf-vs-sf` — generates `n` Stockfish vs Stockfish games at configurable skill levels.
- `POST /api/db/generate-reiLea-vs-sf` — generates `n` games of ReiLea against Stockfish with configurable color assignment and skill level.

The static directory is mounted at `/static` and `GET /` serves `static/index.html`.

To start the server: `uvicorn server:app --reload`

---

### static/index.html

This is a self-contained single-page application. It has no build step and no external JavaScript dependencies beyond Google Fonts. All rendering is done by vanilla JavaScript manipulating the DOM.

The page has two views: the main menu and the game board. On the menu the user picks a game mode (human vs human, play white against a bot, or play black against a bot). The board view shows the chess board, a captured-piece display above and below, a status bar, a scrollable move list in SAN notation, and buttons for resigning, flipping the board, toggling the PGN display, downloading the PGN, starting a new game, and returning to the menu.

Board state is maintained client-side as a 64-element array parsed from the FEN string returned by the API. Click handling works by tracking a selected square and filtering the legal moves returned by the server to determine valid destinations. When a pawn reaches the back rank a promotion modal appears. En passant captures are detected by checking if a pawn moved diagonally onto an empty square.

When it is the bot's turn the frontend calls `POST /api/game/{game_id}/bot-move` and applies the result. The bot response is triggered automatically after every human move in the appropriate modes, with a short timeout to give visual feedback before the move appears.

---

### train.py

This file closes the learning loop between live gameplay and the neural network. It is the component that makes ReiLea actually improve from playing Stockfish.

`ReiLeaTrainer` is instantiated once at server startup and holds a reference to the same `ReiLeaAgent` and `GameDatabase` objects used by `server.py`. It tracks how many `reiLea-vs-sf` games have been played since the last training run via an in-memory counter.

**Trigger:** every 50 `reiLea-vs-sf` games (configurable via `BATCH_TRIGGER`), the trainer fires automatically. It runs in a background thread via `asyncio.run_in_executor` so the FastAPI event loop is never blocked.

**Training algorithm — REINFORCE:**

For each position in each of the 200 most recent ReiLea games, the trainer records:
- `state`: the board encoded as an 18-channel 8×8 tensor (via `encode_board` from `reilea.py`)
- `action`: the move that was actually played, as a flat index in [0, 4095] (via `move_to_index`)
- `reward`: +1 if the player who made that move won, -1 if they lost, 0 for draw

Two losses are then computed and backpropagated together:

```
policy_loss = -mean( log_prob(action) * reward )
value_loss  = MSE( value_prediction, reward )
```

The policy loss is the REINFORCE objective: moves made in winning games are reinforced (log-probability increases), moves made in losing games are discouraged (log-probability decreases). The value loss trains the value head to predict game outcomes from board positions, giving ReiLea a sense of who is winning.

After training completes, the updated weights are saved to `models/reiLea_rl.pt`. Because `server.py` holds a reference to the `ReiLeaAgent` whose `model` object was trained in-place, the improved weights are used immediately for the next game — **no server restart is required**.

**API endpoints added by train.py integration:**

- `GET  /api/training/status` — returns `is_training`, `games_since_last_train`, `games_until_next_train`, `last_trained_at`, and the metrics from the last run.
- `POST /api/train` — manually triggers a training run in the background (returns 409 if already training).

The `generate-reiLea-vs-sf` batch endpoint also increments the counter for every game it generates and triggers training if the threshold is crossed.

---

### RL Training/rl_utils.py

This is a standalone data utility for extracting training samples from the PGN database. It is used by the Jupyter notebook but is independent of `server.py` and `reilea.py`.

It defines a simpler 12-channel board encoding (6 piece types times 2 colors, without castling rights or en passant) and two reward functions. The terminal reward assigns +1 to the winner and -1 to the loser. The shaped reward blends 80% terminal outcome with 20% normalized material advantage, which provides a denser learning signal early in training.

`extract_training_data(game)` replays a PGN game move by move, encoding each position before the move is applied and assigning the final result as the reward for every position in the game. It returns a list of dictionaries with keys `state`, `action_uci`, `action_idx`, `reward`, `turn`, `move_num`, and `fen`.

`generate_training_dataset(db)` is a generator that iterates over all games in the database and yields individual samples. It accepts a `max_games` parameter for testing with a subset of data.

Note: this file uses a 12-plane encoding while `reilea.py` uses an 18-plane encoding. The notebook imports `encode_board` and related functions directly from `reilea.py`, so the two encodings do not conflict during actual training.

---

### RL Training/RL engine.ipynb

This notebook runs the full training pipeline for ReiLea. It must be run from the `RL Training/` directory with the parent directory on `sys.path` so it can import from `reilea.py` and `chess_game.py`.

Phase 1 is supervised learning. The notebook loads all games from `game_data/games.pgn`, extracts (board_planes, move_index, value_target) triples using `encode_board` and `move_to_index` from `reilea.py`, splits into train and validation sets, and trains `ReiLeaNet` for 20 epochs. The loss combines cross-entropy on the policy head (predicting which move was played) and mean squared error on the value head (predicting the game outcome from the current player's perspective). The best checkpoint is saved to `models/reiLea_supervised.pt`.

Phase 2 is reinforcement learning using REINFORCE. The notebook starts from the supervised checkpoint and plays ReiLea against Stockfish (Skill 10) for 50 episodes, alternating which color ReiLea plays. For each ReiLea move the notebook records the log-probability and value prediction. After the game ends the reward is computed and a policy gradient update is applied. The loss is: policy loss (negative advantage-weighted log-probability) plus a value loss (MSE between value prediction and actual reward) minus an entropy bonus to encourage exploration. The checkpoint with the highest win rate is saved to `models/reiLea_rl.pt`, which is the file that `server.py` loads at startup.

Phase 3 is evaluation. The notebook runs the saved models against Stockfish at the configured skill level for a set number of games and reports win/draw/loss counts and win rate.

---

## How the Files Interact

```
static/index.html
    |  HTTP REST calls
    v
server.py
    |  imports ChessGame, GameDatabase, STOCKFISH_PATH
    |  imports ReiLeaAgent
    |  imports ReiLeaTrainer
    |  loads models/reiLea_rl.pt at startup
    v
chess_game.py  <----  reilea.py  <----  train.py
    |                    ^                  |
    |  make_reiLea_move(agent)              |  reads game_data/games.pgn
    |  calls agent.predict_move(board)      |  trains reiLea_agent.model in-place
    |                                       |  saves models/reiLea_rl.pt
    v                                       |
game_data/games.pgn  --------------------->+
game_data/index.json (written by GameDatabase.save_game)

          [after every 50 reiLea-vs-sf games]
          server.py calls trainer.train_async()
          train.py runs in background thread
          weights update immediately for next game
          (no server restart needed)

RL Training/RL engine.ipynb
    |  imports from reilea.py (encode_board, ReiLeaNet, ReiLeaAgent, ...)
    |  imports from chess_game.py (GameDatabase, STOCKFISH_PATH)
    |  reads game_data/ for training data
    |  writes models/reiLea_supervised.pt and models/reiLea_rl.pt

RL Training/rl_utils.py
    |  imports GameDatabase from chess_game.py
    |  used as a helper by the notebook for data extraction
```

The flow of a typical session is:

1. The user starts the server with `uvicorn server:app --reload`. The server loads `models/reiLea_rl.pt` into a `ReiLeaAgent` instance and opens the `GameDatabase`.
2. The user opens `http://localhost:8000` in a browser, which serves `static/index.html`.
3. The user selects a game mode. The frontend posts to `/api/game/new`, receiving a game ID and the starting position.
4. On each human move the frontend posts to `/api/game/{id}/move`. On each bot turn it posts to `/api/game/{id}/bot-move`. The server calls the appropriate method on the `ChessGame` object.
5. When the game ends the server calls `GameDatabase.save_game()`, which appends the PGN to `game_data/games.pgn` and updates `game_data/index.json`. The active game is removed from memory.
6. If the game was a `reiLea-vs-sf` game, `server.py` increments the trainer's counter. When the counter reaches 50, `ReiLeaTrainer.train_async()` is called: training runs in a background thread, the model weights are updated in-place, the improved model is saved to `models/reiLea_rl.pt`, and ReiLea plays stronger in the very next game — no restart required.
7. Training status (is it running, how many games until the next run, last metrics) can be checked at `GET /api/training/status` or triggered manually via `POST /api/train`.
8. For offline or initial training the operator can still run the notebook in `RL Training/`. It reads `game_data/games.pgn`, runs supervised then reinforcement learning, and writes `models/reiLea_supervised.pt` and `models/reiLea_rl.pt`.

---

## Dependencies

```
python-chess
fastapi
uvicorn
pydantic
torch
numpy
```

Stockfish must be installed separately and available on the system PATH, or its binary path must resolve correctly via `shutil.which("stockfish")`.

---

## Quick Start

```bash
pip install python-chess fastapi uvicorn pydantic torch numpy

# Start the server
uvicorn server:app --reload

# Open the UI
# Navigate to http://localhost:8000

# Import external games (optional, for training data)
python -c "
from chess_game import GameDatabase
db = GameDatabase()
count = db.import_pgn('lichess_games.pgn')
print(f'Imported {count} games')
"

# Train ReiLea (from RL Training/)
cd "RL Training"
jupyter notebook "RL engine.ipynb"
```
