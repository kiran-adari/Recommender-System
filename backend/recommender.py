import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests


# ---------- Helpers: load data ----------

def load_ratings(data_path: str = "u.data") -> pd.DataFrame:
    """
    Load MovieLens ratings from u.data.
    Returns a DataFrame with: user_id, item_id, rating
    """
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df[["user_id", "item_id", "rating"]]


def load_movie_titles(path: str = "u.item") -> Dict[int, str]:
    """
    Load MovieLens movie titles from u.item.

    Returns a dict: { item_id (int) -> title (str) }
    """
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],  # only movie id and title
    )
    df.columns = ["item_id", "title"]
    return {int(row.item_id): str(row.title) for _, row in df.iterrows()}


# ---------- TMDB poster helper ----------

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
_POSTER_CACHE: dict = {}  # title -> url


def fetch_poster_url(title: str) -> str:
    """
    Look up a poster URL for this movie title using TMDB.
    Results are cached so we don't call TMDB repeatedly.
    We also strip the year in parentheses, e.g. "Toy Story (1995)" -> "Toy Story".
    """
    if not TMDB_API_KEY or not title:
        return ""

    # Remove year in parentheses: "Cyclo (1995)" -> "Cyclo"
    clean_title = title.rsplit("(", 1)[0].strip()

    cache_key = clean_title.lower()
    if cache_key in _POSTER_CACHE:
        return _POSTER_CACHE[cache_key]

    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": TMDB_API_KEY,
                "query": clean_title,
                "include_adult": "false",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            _POSTER_CACHE[cache_key] = ""
            return ""

        poster_path = results[0].get("poster_path")
        if not poster_path:
            _POSTER_CACHE[cache_key] = ""
            return ""

        url = f"https://image.tmdb.org/t/p/w342{poster_path}"
        _POSTER_CACHE[cache_key] = url
        return url
    except Exception:
        _POSTER_CACHE[cache_key] = ""
        return ""


# ---------- Core CF model ----------

class ItemItemCFModel:
    """
    Item-item collaborative filtering model built from a ratings DataFrame.

    - Builds a user × item rating matrix R.
    - Computes item-item cosine similarity.
    - Can recommend items for a given user_id.
    """

    def __init__(self, ratings_df: pd.DataFrame, movie_titles: Optional[dict] = None):
        ratings = ratings_df.copy()
        self.ratings = ratings
        self.movie_titles = movie_titles or {}

        # ----- Build ID mappings -----
        users = sorted(ratings["user_id"].unique())
        items = sorted(ratings["item_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}

        self.item_to_idx = {m: j for j, m in enumerate(items)}
        self.idx_to_item = {j: m for m, j in self.item_to_idx.items()}

        n_users = len(users)
        n_items = len(items)
        self.n_items = n_items

        # ----- Build user-item matrix R (users × items) -----
        R = np.zeros((n_users, n_items), dtype=np.float32)
        for row in ratings.itertuples(index=False):
            u = self.user_to_idx[row.user_id]
            i = self.item_to_idx[row.item_id]
            R[u, i] = row.rating

        self.R = R  # shape (n_users, n_items)

        # ----- Compute item-item cosine similarity -----
        # Each column is an item vector (all users' ratings for that item)
        item_vectors = R  # (n_users, n_items)

        # Norm of each item vector
        norms = np.linalg.norm(item_vectors, axis=0)  # length n_items
        norms[norms == 0] = 1e-8  # avoid divide-by-zero

        # Similarity matrix S = (X^T X) / (||xi|| * ||xj||)
        S = item_vectors.T @ item_vectors  # items × items
        S = S / (norms[None, :] * norms[:, None])
        S = np.clip(S, -1.0, 1.0)  # keep within [-1, 1]

        self.sim = S

        # ----- Popularity backup (for cold-start) -----
        movie_stats = (
            ratings.groupby("item_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )
        movie_stats = movie_stats[movie_stats["count"] >= 20].copy()
        movie_stats = movie_stats.sort_values(
            by=["mean", "count"], ascending=[False, False]
        )

        popular_items = []
        for _, row in movie_stats.iterrows():
            item_id = int(row["item_id"])
            title = self.movie_titles.get(item_id, "")
            poster_url = fetch_poster_url(title)
            popular_items.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "poster_url": poster_url,
                    "score": float(row["mean"]),
                }
            )
        self.popular_items = popular_items

    def popularity_recs(self, top_k: int = 5) -> List[dict]:
        return self.popular_items[:top_k]

    def recommend(self, user_id: int, top_k: int = 5) -> List[dict]:
        """
        Recommend top_k items for this user.
        If the user is unknown or has no ratings, fall back to popularity.
        """
        if user_id not in self.user_to_idx:
            return self.popularity_recs(top_k=top_k)

        u = self.user_to_idx[user_id]
        user_ratings = self.R[u, :]  # shape (n_items,)

        rated_mask = user_ratings > 0
        if not rated_mask.any():
            return self.popularity_recs(top_k=top_k)

        scores = np.zeros(self.n_items, dtype=np.float32)

        # For each item j the user has NOT rated, predict a score
        for j in range(self.n_items):
            if rated_mask[j]:
                continue  # skip items already rated

            sims = self.sim[j, rated_mask]     # similarities
            r = user_ratings[rated_mask]       # user ratings

            if np.all(sims == 0):
                continue

            num = float(np.dot(sims, r))
            denom = float(np.sum(np.abs(sims)) + 1e-8)
            scores[j] = num / denom

        # Use only positive scores as candidates
        candidate_indices = np.where(scores > 0)[0]
        if len(candidate_indices) == 0:
            return self.popularity_recs(top_k=top_k)

        candidate_scores = scores[candidate_indices]
        order = np.argsort(-candidate_scores)  # sort desc
        top_indices = candidate_indices[order][:top_k]

        recs: List[dict] = []
        for j in top_indices:
            item_id = self.idx_to_item[j]
            title = self.movie_titles.get(item_id, "")
            poster_url = fetch_poster_url(title)
            recs.append(
                {
                    "item_id": int(item_id),
                    "title": title,
                    "poster_url": poster_url,
                    "score": float(scores[j]),
                }
            )
        return recs


# ---------- Attack + Defense helpers ----------

def simulate_attack(
    ratings_df: pd.DataFrame,
    attacker_id: int,
    target_item_id: int,
    n_push_items: int = 10,
    extreme_rating: float = 5.0,
) -> pd.DataFrame:
    """
    Simple 'push' attack:
    - Add a fake user (attacker_id)
    - Give the target item a very high rating
    - Give some other items also high ratings
    """
    rng = np.random.default_rng(42)
    df = ratings_df.copy()

    all_items = df["item_id"].unique()
    candidate_items = [i for i in all_items if i != target_item_id]

    if len(candidate_items) > n_push_items:
        push_items = rng.choice(candidate_items, size=n_push_items, replace=False)
    else:
        push_items = candidate_items

    new_rows = []

    # Attack the target item
    new_rows.append(
        {"user_id": attacker_id, "item_id": target_item_id, "rating": extreme_rating}
    )

    # Give high ratings to some other items
    for it in push_items:
        new_rows.append(
            {"user_id": attacker_id, "item_id": int(it), "rating": extreme_rating}
        )

    if new_rows:
        df_attack = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_attack = df

    return df_attack


def apply_defense(ratings_df: pd.DataFrame, tau: float = 1.5) -> pd.DataFrame:
    """
    Simple statistical defense:

    For each user:
      - Compute mean and std of their ratings.
      - Clip any rating that is more than tau * std away from the mean.
    """

    def clip_user(group: pd.DataFrame) -> pd.DataFrame:
        ratings = group["rating"].values.astype(float)
        mu = ratings.mean()
        sigma = ratings.std()
        if sigma < 1e-6:
            return group

        lower = mu - tau * sigma
        upper = mu + tau * sigma
        clipped = np.clip(ratings, lower, upper)
        group = group.copy()
        group["rating"] = clipped
        return group

    defended = ratings_df.groupby("user_id", group_keys=False).apply(clip_user)
    return defended.reset_index(drop=True)


# ---------- Scenario wrapper ----------

class ScenarioRecommender:
    """
    Builds three models:

      - baseline: trained on original ratings
      - attack:   trained on ratings + a fake attacker user
      - defense:  trained on attacked ratings after clipping outliers

    Frontend chooses which one by sending scenario = 'baseline' / 'attack' / 'defense'.
    """

    def __init__(self, data_path: str = "u.data", movie_path: str = "u.item"):
        # Load clean ratings and movie titles
        base_ratings = load_ratings(data_path)
        movie_titles = load_movie_titles(movie_path)

        self.base_ratings = base_ratings
        self.movie_titles = movie_titles

        # Baseline model
        self.baseline_model = ItemItemCFModel(base_ratings, movie_titles)

        # Choose an attacker id (new fake user) and a target item
        rng = np.random.default_rng(42)
        self.attacker_id = int(base_ratings["user_id"].max() + 1)

        item_counts = base_ratings["item_id"].value_counts()
        popular_items = item_counts[item_counts >= 50].index
        if len(popular_items) == 0:
            popular_items = item_counts.index
        self.target_item_id = int(rng.choice(popular_items))

        # Attack model
        attack_ratings = simulate_attack(
            base_ratings,
            attacker_id=self.attacker_id,
            target_item_id=self.target_item_id,
            n_push_items=10,
            extreme_rating=5.0,
        )
        self.attack_ratings = attack_ratings
        self.attack_model = ItemItemCFModel(attack_ratings, movie_titles)

        # Defense model
        defended_ratings = apply_defense(attack_ratings, tau=1.5)
        self.defense_ratings = defended_ratings
        self.defense_model = ItemItemCFModel(defended_ratings, movie_titles)

    def recommend(self, scenario: str, user_id: int, top_k: int = 5) -> List[dict]:
        scenario = (scenario or "").lower()
        if scenario == "attack":
            model = self.attack_model
        elif scenario == "defense":
            model = self.defense_model
        else:
            model = self.baseline_model

        return model.recommend(user_id=user_id, top_k=top_k)