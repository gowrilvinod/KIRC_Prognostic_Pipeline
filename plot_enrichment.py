import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
df = pd.read_csv("./aggregated_results/kirc_pathway_enrichment_expanded.csv")

# Create figures directory if it doesn't exist
Path("./figures").mkdir(parents=True, exist_ok=True)

# Function to parse overlap count
def get_overlap_count(overlap_str):
    if pd.isna(overlap_str):
        return 0
    return len(str(overlap_str).split(", "))

df["Overlap"] = df["Overlapping Genes"].apply(get_overlap_count)
df["log_FDR"] = -np.log10(df["Adjusted P-value (FDR)"] + 1e-15) # Avoid log(0)

# Set style (standard matplotlib)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

def plot_cohort_library(cohort_name, library_name, title_suffix, filename):
    sub = df[(df["Cohort"] == cohort_name) & (df["Library"] == library_name)].copy()
    if sub.empty:
        print(f"No results for {cohort_name} - {library_name}")
        return
        
    # Take top 10 by P-value
    sub = sub.sort_values("P-value").head(10)
    
    # Sort for plotting (most significant at the top)
    sub = sub.sort_values("P-value", ascending=False)
    
    # Clean term names (e.g. remove GO IDs for cleaner plots)
    sub["Term_Clean"] = sub["Term"].apply(lambda x: x.split(" (GO:")[0])
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Scatter plot
    scatter = ax.scatter(
        x=sub["Combined Score"],
        y=sub["Term_Clean"],
        s=sub["Overlap"] * 45, # Scale bubble size
        c=sub["log_FDR"],
        cmap="plasma",
        alpha=0.85,
        edgecolors="grey",
        linewidths=0.5
    )
    
    # Titles and labels
    ax.set_title(f"Top 10 Enriched {title_suffix} Pathways\n({cohort_name.replace('_', ' ')})", pad=15)
    ax.set_xlabel("Combined Score")
    ax.set_ylabel("")
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label("-log10(Adjusted P-value / FDR)")
    
    # Legend for sizes (Overlap count)
    sizes = sorted(sub["Overlap"].unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(s),
                                  markerfacecolor='grey', markersize=np.sqrt(s * 45),
                                  alpha=0.6) for s in sizes]
    ax.legend(handles=legend_elements, title="Overlapping Genes", loc="lower right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"./figures/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: ./figures/{filename}")

# Generate plots
cohorts = ["KIRC_Male_Top300", "KIRC_Female_Top300"]
libraries = [
    ("GO_Biological_Process_2023", "GO BP", "go_bp"),
    ("KEGG_2021_Human", "KEGG", "kegg")
]

for cohort in cohorts:
    for lib_name, suffix, file_prefix in libraries:
        fn = f"{cohort.lower()}_{file_prefix}_enrichment.png"
        plot_cohort_library(cohort, lib_name, suffix, fn)
