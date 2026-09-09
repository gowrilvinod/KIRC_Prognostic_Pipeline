import pandas as pd
import re
from pathlib import Path

# 1. Load the expanded pathway enrichment CSV
csv_path = Path("./aggregated_results/kirc_pathway_enrichment_expanded.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Source file not found: {csv_path}")

df = pd.read_csv(csv_path)

# 2. Filter for significant entries (FDR < 0.05) and save to a new CSV
df_sig = df[df["Adjusted P-value (FDR)"] < 0.05].copy()
output_csv = Path("./aggregated_results/kirc_pathway_enrichment_significant.csv")
df_sig.to_csv(output_csv, index=False)
print(f"Significant-only results (FDR < 0.05) saved to: {output_csv}\n")

# Helper function to extract ID from Term string
def parse_term_and_id(term):
    term_name = term
    term_id = "N/A"
    # Match trailing parentheses containing digits or specific database prefixes
    match = re.search(r'\(([^)]+)\)$', term)
    if match:
        possible_id = match.group(1)
        if "GO:" in possible_id or "hsa" in possible_id or any(c.isdigit() for c in possible_id):
            term_id = possible_id
            term_name = term[:match.start()].strip()
    return term_name, term_id

# Helper function to print reports for a group
def print_report_section(df_source, cohort_prefix, library_name, label):
    # Filter for the specific cohort (Top300/Top500) and library (GO/KEGG)
    cohort_df = df_source[df_source["Cohort"] == cohort_prefix]
    lib_df = cohort_df[cohort_df["Library"] == library_name]
    
    # Filter for FDR < 0.05
    sig_df = lib_df[lib_df["Adjusted P-value (FDR)"] < 0.05]
    
    print(f"  {label}:")
    if sig_df.empty:
        print("    No significant enrichment (FDR < 0.05)")
        print()
        return
        
    for idx, row in sig_df.iterrows():
        clean_name, term_id = parse_term_and_id(row["Term"])
        print(f"    - Term: {clean_name}")
        print(f"      ID: {term_id}")
        print(f"      Raw P-value: {row['P-value']:.4e}")
        print(f"      FDR: {row['Adjusted P-value (FDR)']:.4e}")
        print(f"      Overlapping Genes: {row['Overlapping Genes']}")
        print(f"      Combined Score: {row['Combined Score']:.4f}")
        print()

# 3. Print the formatted terminal output
# ======================================================================
# KIRC MALE COHORT
# ======================================================================
print("=" * 70)
print("KIRC MALE COHORT")
print("=" * 70)
print()

print("[Top300 Analysis]")
print_report_section(df, "KIRC_Male_Top300", "GO_Biological_Process_2023", "GO Biological Process")
print_report_section(df, "KIRC_Male_Top300", "KEGG_2021_Human", "KEGG")

print("[Top500 Analysis]")
print_report_section(df, "KIRC_Male_Top500", "GO_Biological_Process_2023", "GO Biological Process")
print_report_section(df, "KIRC_Male_Top500", "KEGG_2021_Human", "KEGG")

# ======================================================================
# KIRC FEMALE COHORT
# ======================================================================
print("=" * 70)
print("KIRC FEMALE COHORT")
print("=" * 70)
print()

print("[Top300 Analysis]")
print_report_section(df, "KIRC_Female_Top300", "GO_Biological_Process_2023", "GO Biological Process")
print_report_section(df, "KIRC_Female_Top300", "KEGG_2021_Human", "KEGG")

print("[Top500 Analysis]")
print_report_section(df, "KIRC_Female_Top500", "GO_Biological_Process_2023", "GO Biological Process")
print_report_section(df, "KIRC_Female_Top500", "KEGG_2021_Human", "KEGG")
