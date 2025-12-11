# attack_experiment.py
"""
Run a simple experiment to see how the attack and defense
change the rank of the target movie for some users.
"""

from recommender import ScenarioRecommender


def find_rank_of_item(recs, target_item_id):
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

    print("=== Attack Experiment ===")
    print(f"Target item ID: {target_id}")
    print(f"Target title : {target_title}")
    print()

    # pick a few user IDs to inspect
    user_ids = [1, 10, 50, 100, 150]

    for uid in user_ids:
        print(f"--- User {uid} ---")

        # Baseline recommendations
        base_recs = scen.baseline_model.recommend(user_id=uid, top_k=50)
        base_rank, base_score = find_rank_of_item(base_recs, target_id)

        # Attack recommendations
        atk_recs = scen.attack_model.recommend(user_id=uid, top_k=50)
        atk_rank, atk_score = find_rank_of_item(atk_recs, target_id)

        # Defense recommendations
        def_recs = scen.defense_model.recommend(user_id=uid, top_k=50)
        def_rank, def_score = find_rank_of_item(def_recs, target_id)

        print(f"Baseline: rank={base_rank}, score={base_score}")
        print(f"Attack  : rank={atk_rank}, score={atk_score}")
        print(f"Defense : rank={def_rank}, score={def_score}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()