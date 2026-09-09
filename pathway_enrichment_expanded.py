import urllib.request
import urllib.parse
import json
import subprocess
import pandas as pd
from pathlib import Path

# Load RRA raw rankings
male_df = pd.read_csv("./aggregated_results/male_expr_raw_rra.csv")
female_df = pd.read_csv("./aggregated_results/female_expr_raw_rra.csv")

# Filter out empty or null names
male_genes_all = male_df["Name"].dropna().tolist()
female_genes_all = female_df["Name"].dropna().tolist()

def run_enrichment(genes, description):
    print(f"Running enrichment for {description} ({len(genes)} genes)...")
    
    # Write genes to a temp file for curl upload
    temp_file = Path("temp_genes.txt")
    with open(temp_file, "w") as f:
        f.write('\n'.join(genes))
        
    # 1. Add list to Enrichr using curl (multipart/form-data)
    cmd = [
        "curl", "-s",
        "-F", f"list=@{temp_file}",
        "-F", f"description={description}",
        "https://maayanlab.cloud/Enrichr/addList"
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        res_data = json.loads(res.stdout)
        user_list_id = res_data["userListId"]
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise Exception(f"Failed to submit to Enrichr via curl: {e}")
        
    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()

    # 2. Query enrichment
    enrich_base_url = "https://maayanlab.cloud/Enrichr/enrich"
    libraries = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
    all_results = []
    
    for lib in libraries:
        params = {
            'userListId': user_list_id,
            'backgroundType': lib
        }
        encoded_params = urllib.parse.urlencode(params)
        enrich_url = f"{enrich_base_url}?{encoded_params}"
        
        try:
            with urllib.request.urlopen(enrich_url) as res:
                res_data = json.loads(res.read().decode('utf-8'))
                if lib in res_data:
                    terms = res_data[lib]
                    for t in terms[:25]: # Save top 25 terms
                        all_results.append({
                            "Cohort": description,
                            "Library": lib,
                            "Term": t[1],
                            "P-value": t[2],
                            "Adjusted P-value (FDR)": t[6],
                            "Combined Score": t[4],
                            "Overlapping Genes": ", ".join(t[5])
                        })
        except Exception as e:
            print(f"Warning: Failed to fetch {lib} results: {e}")
            continue
            
    return all_results

# Execute for Top 300 and Top 500
try:
    results = []
    for size in [300, 500]:
        results.extend(run_enrichment(male_genes_all[:size], f"KIRC_Male_Top{size}"))
        results.extend(run_enrichment(female_genes_all[:size], f"KIRC_Female_Top{size}"))

    df = pd.DataFrame(results)
    df.to_csv("./aggregated_results/kirc_pathway_enrichment_expanded.csv", index=False)
    print("\n=== Expanded Pathway Enrichment Completed and Saved! ===")
    print(df.head(20))
except Exception as e:
    print(f"Error during enrichment: {e}")
