# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import subprocess
import sys

from src.models.predict_model import recommend_similar_movies

app = FastAPI(title="Movie Recommendation API", version="1.0.0")


# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    movie_id: int = Field(..., ge=1, description="movieId de référence")
    top_n: int = Field(10, ge=1, le=50, description="Nombre de recommandations")


class TrainingRequest(BaseModel):
    force: bool = Field(
        False,
        description="Si true, lance l'entraînement même si des artefacts existent déjà",
    )


# -----------------------------
# Endpoint PREDICT
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return recommend_similar_movies(movie_id=req.movie_id, top_n=req.top_n)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# -----------------------------
# Endpoint TRAINING
# -----------------------------
@app.post("/training")
def training(req: TrainingRequest = TrainingRequest()):
    """
    Lance le réentraînement via le script existant:
      python src/models/train_model.py

    On renvoie stdout/stderr pour que tu voies le résultat dans Swagger.
    """
    cmd = [sys.executable, "src/models/train_model.py"]

    # Optionnel: si ton train_model.py gère --force, on passe l'arg.
    # Sinon, ce flag sera ignoré (mais ça ne casse pas).
    if req.force:
        cmd.append("--force")

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "status": "ok",
            "message": "Model retrained successfully",
            "stdout": completed.stdout,
        }
    except subprocess.CalledProcessError as e:
        # e.stdout / e.stderr existent si capture_output=True
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Training failed",
                "returncode": e.returncode,
                "stdout": e.stdout,
                "stderr": e.stderr,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
