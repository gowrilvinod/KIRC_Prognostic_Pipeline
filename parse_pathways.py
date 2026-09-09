import pandas as pd
from pathlib import Path

def print_summary(csv_path, title):
    p = Path(csv_path)
    if not p.exists():
        print(f"\n[Warning] File not found: {csv_path}")
        return
        
    df = pd.read_csv(p)
    print(f"\n========================================================")
    print(f"=== {title} ===")
    print(f"========================================================")
    
    for cohort in df["Cohort"].unique():
        print(f"\n[{cohort}]")
        sub = df[df["Cohort"] == cohort]
        for lib in sub["Library"].unique():
            lib_short = "GO_BP" if "GO" in lib else "KEGG"
            sig = sub[(sub["Library"] == lib) & (sub["Adjusted P-value (FDR)"] < 0.05)]
            if not sig.empty:
                print(f"  {lib_short}: {len(sig)} significant. Top 3:")
                for idx, row in sig.head(3).iterrows():
                    print(f"   * {row['Term']} (FDR: {row['Adjusted P-value (FDR)']:.3e})")
            else:
                print(f"  {lib_short}: None significant. Top 3 by raw P:")
                top_3 = sub[sub["Library"] == lib].sort_values("P-value").head(3)
                for idx, row in top_3.iterrows():
                    print(f"   * {row['Term']} (p: {row['P-value']:.3e})")

# Print summaries for both runs
print_summary("./aggregated_results/kirc_pathway_enrichment.csv", "16-GENE (8 MALE / 8 FEMALE) SIGNATURE RESULTS")
print_summary("./aggregated_results/kirc_pathway_enrichment_expanded.csv", "EXPANDED 300/500-GENE RESULTS")
