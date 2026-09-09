import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import beta

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
        # Create lookup for O(1) rank search
        rank_lookup = {item: idx+1 for idx, item in enumerate(r_list)}
        
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

def main():
    print("=== Running Robust Rank Aggregation in Python ===")
    
    folder = os.path.expanduser("~/spring_2026/results_folded_1")
    search_path = os.path.join(folder, "*.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No result CSV files found in {folder}. Please run modeling first.")
        return
        
    print(f"Found {len(files)} rank files in results directory.")
    
    # Separate by sex
    male_files = [f for f in files if "male" in os.path.basename(f).lower() and "female" not in os.path.basename(f).lower()]
    female_files = [f for f in files if "female" in os.path.basename(f).lower()]
    
    print(f"Male files: {len(male_files)}, Female files: {len(female_files)}")
    
    for sex, files_subset in [("male", male_files), ("female", female_files)]:
        print(f"\nProcessing rankings for {sex}...")
        rank_lists = []
        all_unique_items = set()
        
        for file in files_subset:
            df = pd.read_csv(file)
            
            # Split by voter
            groups = df.groupby("Voter")
            for voter, group in groups:
                # Sort by Rank and get item list
                sorted_group = group.sort_values(by="Rank")
                voter_items = sorted_group["ItemID"].tolist()
                rank_lists.append(voter_items)
                all_unique_items.update(voter_items)
                
        print(f"Extracted {len(rank_lists)} ranking voter lists for {sex} (unique features = {len(all_unique_items)}).")
        
        # Run RRA
        rra_results = robust_rank_aggregation(rank_lists, list(all_unique_items))
        
        # Add consensus rank
        rra_results["ConsensusRank"] = range(1, len(rra_results) + 1)
        
        # Filter for significant features (Score < 0.05)
        sig_df = rra_results[rra_results["Score"] < 0.05]
        
        # Save raw results temporarily to load for overlap removal
        out_dir = "./aggregated_results"
        os.makedirs(out_dir, exist_ok=True)
        rra_results.to_csv(f"{out_dir}/{sex}_expr_raw_rra.csv", index=False)
        print(f"Saved raw RRA rankings with scores to {out_dir}/{sex}_expr_raw_rra.csv. Significant features: {len(sig_df)}")

    # 4. Overlap Removal between Male and Female
    male_raw = pd.read_csv("./aggregated_results/male_expr_raw_rra.csv")
    female_raw = pd.read_csv("./aggregated_results/female_expr_raw_rra.csv")
    
    male_sig_genes = set(male_raw[male_raw["Score"] < 0.05]["Name"].tolist())
    female_sig_genes = set(female_raw[female_raw["Score"] < 0.05]["Name"].tolist())
    
    overlap = male_sig_genes & female_sig_genes
    print(f"\nSignificant Male genes: {len(male_sig_genes)}")
    print(f"Significant Female genes: {len(female_sig_genes)}")
    print(f"Overlapping significant genes (to exclude): {len(overlap)}: {list(overlap)}")
    
    male_only = male_sig_genes - overlap
    female_only = female_sig_genes - overlap
    
    male_only_df = male_raw[male_raw["Name"].isin(male_only)][["Name", "ConsensusRank", "Score"]]
    female_only_df = female_raw[female_raw["Name"].isin(female_only)][["Name", "ConsensusRank", "Score"]]
    
    # Save final sex-specific rankings
    male_only_df.to_csv("./aggregated_results/male_expr_aggregated_ranking.csv", index=False)
    female_only_df.to_csv("./aggregated_results/female_expr_aggregated_ranking.csv", index=False)
    
    print(f"Saved final sex-specific male ranking to ./aggregated_results/male_expr_aggregated_ranking.csv (size: {len(male_only_df)})")
    print(f"Saved final sex-specific female ranking to ./aggregated_results/female_expr_aggregated_ranking.csv (size: {len(female_only_df)})")

if __name__ == "__main__":
    main()
