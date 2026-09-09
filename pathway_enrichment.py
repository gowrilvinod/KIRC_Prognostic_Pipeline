import urllib.request
import urllib.parse
import json
import subprocess
import pandas as pd
from pathlib import Path

# Final 8-gene consensus signatures
male_genes = ["HS3ST1", "RNF183", "LINC00973", "HBB", "CDKN1A", "HMGA2", "HS3ST3A1", "ADAM8"]
female_genes = ["SNTG2-AS1", "SSTR1", "ACP5", "RNY4P34", "TNFSF15", "NUDT2", "ACTN2", "CTTNBP2"]

def run_enrichment(genes, description):
    print(f"Running enrichment for {description} (8 genes)...")
    temp_file = Path("temp_genes_8.txt")
    with open(temp_file, "w") as f:
        f.write('\n'.join(genes))
        
    # Submit list via curl
    cmd = [
        "curl", "-s",
        "-F", f"list=@{temp_file}",
        "-F", f"description={description}",
        "https://maayanlab.cloud/Enrichr/addList"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    user_list_id = json.loads(res.stdout)["userListId"]
    temp_file.unlink()

    # Query terms
    enrich_base_url = "https://maayanlab.cloud/Enrichr/enrich"
    libraries = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
    all_results = []
    
    for lib in libraries:
        params = {
            'userListId': user_list_id,
            'backgroundType': lib
        }
        enrich_url = f"{enrich_base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(enrich_url) as res:
            res_data = json.loads(res.read().decode('utf-8'))
            if lib in res_data:
                terms = res_data[lib]
                for t in terms[:10]: # Save top 10 terms
                    all_results.append({
                        "Cohort": description,
                        "Library": lib,
                        "Term": t[1],
                        "P-value": t[2],
                        "Adjusted P-value (FDR)": t[6],
                        "Combined Score": t[4],
                        "Overlapping Genes": ", ".join(t[5])
                    })
    return all_results

try:
    results = []
    results.extend(run_enrichment(male_genes, "KIRC_Male_8"))
    results.extend(run_enrichment(female_genes, "KIRC_Female_8"))

    df = pd.DataFrame(results)
    df.to_csv("./aggregated_results/kirc_pathway_enrichment.csv", index=False)
    print("\n=== 8-Gene Pathway Enrichment Completed and Saved! ===")
    print(df.head(10))
except Exception as e:
    print(f"Error during enrichment: {e}")
