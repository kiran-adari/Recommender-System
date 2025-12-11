# metrics_experiment.py
"""
Compute simple metrics for baseline vs attack vs defense:
- Hit@k for the target movie
- Average rank (when it appears in top-k)

You will use these numbers in your report.
"""

from statistics import mean
from typing import List, Tuple, Optional

from recommender import ScenarioRecommender


def find_rank_of_item(
    recs: List[dict], target_item_id: int
) -> Tuple[Optional[int], Optional[float]]:
    """
    Given a list of recommendations (list of dicts),
    return (rank_index, score) of the target item if it appears,
    otherwise (None, None).
    """
    for idx, r in enumerate(recs, start=1):
        if r["item_id"] == target_item_id:
            return idx, r["score"]
    return None, None


def main():
    # Build all three models from u.data / u.item
    scen = ScenarioRecommender(data_path="u.data", movie_path="u.item")

    target_id = scen.target_item_id
    target_title = scen.movie_titles.get(target_id, "Unknown title")

    print("=== Metrics Experiment: Baseline vs Attack vs Defense ===")
    print(f"Target item ID : {target_id}")
    print(f"Target title   : {target_title}")
    print()

    # Choose how many users and top-k
    k = 10
    max_users = 200  # you can increase this later if it runs fast

    all_user_ids = sorted(scen.base_ratings["user_id"].unique())
    user_ids = all_user_ids[:max_users]

    print(f"Using top-k = {k}")
    print(f"Number of users in sample = {len(user_ids)}")
    print()

    # Metrics containers
    metrics = {
        "baseline": {"hits": 0, "ranks": []},
        "attack": {"hits": 0, "ranks": []},
        "defense": {"hits": 0, "ranks": []},
    }

    for uid in user_ids:
        # Baseline recs
        base_recs = scen.baseline_model.recommend(user_id=uid, top_k=k)
        base_rank, _ = find_rank_of_item(base_recs, target_id)
        if base_rank is not None:
            metrics["baseline"]["hits"] += 1
            metrics["baseline"]["ranks"].append(base_rank)

        # Attack recs
        atk_recs = scen.attack_model.recommend(user_id=uid, top_k=k)
        atk_rank, _ = find_rank_of_item(atk_recs, target_id)
        if atk_rank is not None:
            metrics["attack"]["hits"] += 1
            metrics["attack"]["ranks"].append(atk_rank)

        # Defense recs
        def_recs = scen.defense_model.recommend(user_id=uid, top_k=k)
        def_rank, _ = find_rank_of_item(def_recs, target_id)
        if def_rank is not None:
            metrics["defense"]["hits"] += 1
            metrics["defense"]["ranks"].append(def_rank)

    # Print summary
    print("=== Summary (over users) ===")
    for name in ["baseline", "attack", "defense"]:
        hits = metrics[name]["hits"]
        ranks = metrics[name]["ranks"]
        hit_rate = hits / len(user_ids)

        if ranks:
            avg_rank = mean(ranks)
        else:
            avg_rank = None

        print(f"\nScenario: {name}")
        print(f"  Hit@{k} (target appears in top-{k}): {hits} / {len(user_ids)}")
        print(f"  Hit@{k} rate                       : {hit_rate:.4f}")
        if avg_rank is not None:
            print(f"  Avg. rank of target (smaller is better): {avg_rank:.2f}")
        else:
            print("  Avg. rank of target: N/A (never appears)")

    print("\nDone.")


if __name__ == "__main__":
    main()
