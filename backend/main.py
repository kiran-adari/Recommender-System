from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommender import ScenarioRecommender

# ---------------- FastAPI app ----------------

app = FastAPI(
    title="Movie Recommender – Baseline / Attack / Defense",
    description="Item–item CF on MovieLens with simulated attack and simple defense.",
    version="1.0.0",
)

# Allow React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for local dev; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load models once ----------------

# Assumes u.data and u.item are in the same folder as this file
scenario_rec = ScenarioRecommender(data_path="u.data", movie_path="u.item")


# ---------------- Request models ----------------

class RecommendRequest(BaseModel):
    scenario: str = "baseline"   # "baseline" | "attack" | "defense"
    user_id: int
    top_k: int = 12


class CompareRequest(BaseModel):
    user_id: int
    top_k: int = 12


# ---------------- Routes ----------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    """
    Single-scenario view – used when mode === 'single' on the frontend.
    """
    scenario = req.scenario.lower()
    if scenario not in {"baseline", "attack", "defense"}:
        scenario = "baseline"

    recs = scenario_rec.recommend(
        scenario=scenario,
        user_id=req.user_id,
        top_k=req.top_k,
    )

    return {
        "user_id": req.user_id,
        "scenario": scenario,
        "top_k": req.top_k,
        "recommendations": recs,
    }


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """
    3-column compare view – used when mode === 'compare' on the frontend.
    Returns top_k recommendations for:
      - baseline
      - attack
      - defense
    """
    baseline = scenario_rec.recommend("baseline", req.user_id, req.top_k)
    attack = scenario_rec.recommend("attack", req.user_id, req.top_k)
    defense = scenario_rec.recommend("defense", req.user_id, req.top_k)

    return {
        "user_id": req.user_id,
        "top_k": req.top_k,
        "baseline": baseline,
        "attack": attack,
        "defense": defense,
    }
