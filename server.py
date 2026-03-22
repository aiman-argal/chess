from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import asyncio
import uuid

from chess_game import ChessGame, GameDatabase
from reilea import ReiLeaAgent
from train import ReiLeaTrainer

app = FastAPI(title="Chess RL", version="1.0.0")

active_games: dict[str, ChessGame] = {}
game_modes: dict[str, str] = {}
db = GameDatabase()

REILEA_MODEL_PATH = "models/reiLea_rl.pt"
reiLea_agent = ReiLeaAgent(model_path=REILEA_MODEL_PATH)

# Stockfish skill level 14 ≈ 1800–2000 Elo (scale 0–20)
SF_SKILL = 14
SF_TIME = 0.1

trainer = ReiLeaTrainer(reiLea_agent, db, REILEA_MODEL_PATH)

BOT_MODES = {"sf-white", "sf-black", "reiLea-white", "reiLea-black", "sf-vs-sf", "reiLea-vs-sf"}


class NewGameRequest(BaseModel):
    white: str = "Human"
    black: str = "Human"
    mode: str = "pvp"


class MoveRequest(BaseModel):
    move: str


def _bot_move_for_game(game: ChessGame, mode: str) -> dict:
    turn = game.board.turn

    if mode == "sf-white" and turn:
        return game.make_stockfish_move(SF_TIME)
    if mode == "sf-black" and not turn:
        return game.make_stockfish_move(SF_TIME)
    if mode == "reiLea-white" and turn:
        return game.make_reiLea_move(reiLea_agent)
    if mode == "reiLea-black" and not turn:
        return game.make_reiLea_move(reiLea_agent)
    if mode == "sf-vs-sf":
        return game.make_stockfish_move(SF_TIME)
    if mode == "reiLea-vs-sf":
        if turn:
            return game.make_reiLea_move(reiLea_agent)
        return game.make_stockfish_move(SF_TIME)
    return {"error": "No bot move applicable for current turn and mode"}


def _cleanup_game(game_id: str, game: ChessGame):
    game.close()
    active_games.pop(game_id, None)
    game_modes.pop(game_id, None)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/game/new")
async def new_game(req: NewGameRequest):
    game_id = str(uuid.uuid4())[:8]
    game = ChessGame(white=req.white, black=req.black, event="Web Game", sf_skill=SF_SKILL)
    active_games[game_id] = game
    game_modes[game_id] = req.mode

    return {
        "game_id": game_id,
        "fen": game.get_fen(),
        "turn": "white",
        "mode": req.mode,
        "legal_moves": game.get_legal_moves(),
    }


@app.post("/api/game/{game_id}/move")
async def make_move(game_id: str, req: MoveRequest):
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    result = game.make_move(req.move)
    if "error" in result:
        raise HTTPException(400, result["error"])

    response = {
        **result,
        "fen": game.get_fen(),
        "legal_moves": game.get_legal_moves(),
        "pgn": game.get_pgn(),
        "move_history": game.move_history,
    }

    if result.get("is_game_over"):
        saved_id = db.save_game(game.game)
        response["saved_game_id"] = saved_id
        _cleanup_game(game_id, game)

    return response


@app.post("/api/game/{game_id}/bot-move")
async def bot_move(game_id: str):
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    mode = game_modes.get(game_id, "pvp")
    result = _bot_move_for_game(game, mode)

    if "error" in result:
        raise HTTPException(400, result["error"])

    response = {
        **result,
        "fen": game.get_fen(),
        "legal_moves": game.get_legal_moves(),
        "pgn": game.get_pgn(),
        "move_history": game.move_history,
    }

    if result.get("is_game_over"):
        saved_id = db.save_game(game.game)
        response["saved_game_id"] = saved_id
        # Count live reiLea-vs-sf games; trigger training after BATCH_TRIGGER
        if mode == "reiLea-vs-sf" and not trainer.is_training:
            if trainer.record_game():
                asyncio.create_task(trainer.train_async())
        _cleanup_game(game_id, game)

    return response


@app.post("/api/game/{game_id}/resign")
async def resign(game_id: str):
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    result_str = game.resign()
    saved_id = db.save_game(game.game)
    pgn = game.get_pgn()
    _cleanup_game(game_id, game)

    return {
        "result": result_str,
        "saved_game_id": saved_id,
        "pgn": pgn,
    }


@app.get("/api/game/{game_id}/state")
async def get_state(game_id: str):
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")

    return {
        "fen": game.get_fen(),
        "turn": "white" if game.board.turn else "black",
        "legal_moves": game.get_legal_moves(),
        "move_history": game.move_history,
        "pgn": game.get_pgn(),
        "is_check": game.board.is_check(),
        "is_game_over": game.board.is_game_over(),
        "mode": game_modes.get(game_id, "pvp"),
    }


@app.get("/api/game/{game_id}/pgn")
async def get_pgn(game_id: str):
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    return PlainTextResponse(game.get_pgn())


@app.get("/api/db/stats")
async def get_db_stats():
    return db.get_stats()


@app.get("/api/db/games")
async def get_db_games(limit: int = 20, offset: int = 0):
    games = db.index.get("games", [])
    return {
        "total": len(games),
        "games": games[offset:offset + limit],
    }


@app.post("/api/db/generate")
async def generate_bot_games(n: int = 10):
    results = []
    for _ in range(n):
        game = ChessGame(white="ReiLea", black="ReiLea", event="Bot Training")
        while not game.board.is_game_over():
            game.make_reiLea_move(reiLea_agent)
        game_id = db.save_game(game.game)
        game.close()
        results.append({
            "game_id": game_id,
            "result": game.game.headers["Result"],
            "moves": len(game.move_history),
        })
    return {"generated": len(results), "games": results}


@app.post("/api/db/generate-sf-vs-sf")
async def generate_sf_vs_sf(n: int = 10, skill_white: int = SF_SKILL, skill_black: int = SF_SKILL):
    import chess.engine as _ce
    from chess_game import STOCKFISH_PATH

    results = []
    for _ in range(n):
        game = ChessGame(white=f"SF-{skill_white}", black=f"SF-{skill_black}", event="SF Training")
        engine_w = _ce.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_b = _ce.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine_w.configure({"Skill Level": skill_white})
        engine_b.configure({"Skill Level": skill_black})

        try:
            while not game.board.is_game_over():
                engine = engine_w if game.board.turn else engine_b
                r = engine.play(game.board, _ce.Limit(time=SF_TIME))
                game.make_move(r.move.uci())
        finally:
            engine_w.quit()
            engine_b.quit()

        game_id = db.save_game(game.game)
        results.append({
            "game_id": game_id,
            "result": game.game.headers["Result"],
            "moves": len(game.move_history),
        })
    return {"generated": len(results), "games": results}


@app.post("/api/db/generate-reiLea-vs-sf")
async def generate_reiLea_vs_sf(n: int = 10, reiLea_color: str = "white", sf_skill: int = SF_SKILL):
    results = []
    for _ in range(n):
        white = "ReiLea" if reiLea_color == "white" else f"Stockfish-{sf_skill}"
        black = f"Stockfish-{sf_skill}" if reiLea_color == "white" else "ReiLea"
        game = ChessGame(white=white, black=black, event="ReiLea vs SF", sf_skill=sf_skill)

        while not game.board.is_game_over():
            is_reiLea_turn = (game.board.turn and reiLea_color == "white") or \
                             (not game.board.turn and reiLea_color == "black")
            if is_reiLea_turn:
                game.make_reiLea_move(reiLea_agent)
            else:
                game.make_stockfish_move(SF_TIME)

        game_id = db.save_game(game.game)
        game.close()
        results.append({
            "game_id": game_id,
            "result": game.game.headers["Result"],
            "moves": len(game.move_history),
        })
        trainer.record_game()

    # Trigger training if batch threshold reached (runs in background)
    if trainer.game_count >= trainer.BATCH_TRIGGER and not trainer.is_training:
        asyncio.create_task(trainer.train_async())

    return {
        "generated": len(results),
        "games": results,
        "training_triggered": trainer.game_count == 0,  # True if reset by trigger
        "games_until_next_train": trainer.status["games_until_next_train"],
    }


@app.get("/api/training/status")
async def training_status():
    return trainer.status


@app.post("/api/train")
async def trigger_training():
    """Manually kick off a training run (e.g. from the UI)."""
    if trainer.is_training:
        raise HTTPException(409, "Training already in progress")
    asyncio.create_task(trainer.train_async())
    return {"message": "Training started in background", "status": trainer.status}


app.mount("/static", StaticFiles(directory="static"), name="static")
