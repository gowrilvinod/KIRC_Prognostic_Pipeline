import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import beta
from pathlib import Path


def robust_rank_aggregation(rank_lists, all_items):
    """
    Implements Kolde's Robust Rank Aggregation (RRA) algorithm.
    - rank_lists: list of lists, where each list is a ranked list of items (most important first).
    - all_items: list of all unique items across all lists.
    """
    m = len(rank_lists)
    if m == 0:
        return pd.DataFrame(columns=["Name", "Score"])

    scores = []

    # 1. Normalize ranks
    normalized_ranks = {}
    for item in all_items:
        normalized_ranks[item] = []

    for r_list in rank_lists:
        L = len(r_list)
        rank_lookup = {item: idx + 1 for idx, item in enumerate(r_list)}

        for item in all_items:
            if item in rank_lookup:
                normalized_ranks[item].append(rank_lookup[item] / L)
            else:
                normalized_ranks[item].append(1.0)

    # 2. Compute minimum Beta probability
    for item in all_items:
        r_vals = sorted(normalized_ranks[item])
        min_p = 1.0

        for k in range(1, m + 1):
            u_k = r_vals[k - 1]
            p_k = beta.cdf(u_k, k, m - k + 1)
            if p_k < min_p:
                min_p = p_k

        # 3. Correct score by multiplying by number of lists (Bonferroni-like correction used in Kolde's RRA)
        adjusted_p = min(1.0, min_p * m)
        scores.append({
            "Name": item,
            "Score": adjusted_p
        })

    result_df = pd.DataFrame(scores).sort_values(by="Score", ascending=True)
    return result_df


def aggregate_single_sex(sex_label):
    print(f"\n--- Aggregating RRA Rankings for CPTAC {sex_label.upper()} Cohort ---")

    folder = Path(f"./aggregated_results/cptac_sex_specific_{sex_label.lower()}_folded_ranks")
    search_pattern = str(folder / f"cptac_sex_specific_{sex_label.lower()}_fold_*.csv")
    files = glob.glob(search_pattern)

    if not files:
        print(f"No CPTAC {sex_label} fold rank CSV files found matching {search_pattern}.")
        return

    print(f"Found {len(files)} CPTAC {sex_label} rank files in {folder}.")

    rank_lists = []
    all_unique_items = set()

    for file in files:
        df = pd.read_csv(file)
        if df.empty or "Voter" not in df.columns:
            continue
        groups = df.groupby("Voter")
        for voter, group in groups:
            sorted_group = group.sort_values(by="Rank")
            voter_items = sorted_group["ItemID"].tolist()
            if voter_items:
                rank_lists.append(voter_items)
                all_unique_items.update(voter_items)

    print(f"[{sex_label.upper()}] Extracted {len(rank_lists)} voter ranking lists (unique features = {len(all_unique_items)}).")

    if not rank_lists:
        print(f"[{sex_label.upper()}] Warning: No valid voter rank lists found.")
        return

    rra_results = robust_rank_aggregation(rank_lists, list(all_unique_items))
    rra_results["ConsensusRank"] = range(1, len(rra_results) + 1)

    out_dir = Path("./aggregated_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"cptac_sex_specific_{sex_label.lower()}_expr_raw_rra.csv"
    rra_results.to_csv(raw_path, index=False)

    sig_df = rra_results[rra_results["Score"] < 0.05]
    print(f"[{sex_label.upper()}] Saved raw RRA rankings to {raw_path}. Total features: {len(rra_results)}, Significant features (p < 0.05): {len(sig_df)}")

    final_path = out_dir / f"cptac_sex_specific_{sex_label.lower()}_expr_aggregated_ranking.csv"
    sig_df[["Name", "ConsensusRank", "Score"]].to_csv(final_path, index=False)
    print(f"[{sex_label.upper()}] Saved final aggregated ranking to {final_path} (size: {len(sig_df)})")


def main():
    print("=== Running Robust Rank Aggregation for CPTAC Sex-Specific Discovery ===")
    aggregate_single_sex("male")
    aggregate_single_sex("female")


if __name__ == "__main__":
    main()
