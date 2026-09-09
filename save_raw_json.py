import urllib.request
import urllib.parse
import json
import subprocess
import pandas as pd
from pathlib import Path

# Load raw rankings locally
male_df = pd.read_csv("./aggregated_results/male_expr_raw_rra.csv")
female_df = pd.read_csv("./aggregated_results/female_expr_raw_rra.csv")

male_genes = male_df["Name"].dropna().head(300).tolist()
female_genes = female_df["Name"].dropna().head(300).tolist()

def save_raw_enrichr(genes, description, prefix):
    # 1. Add list using curl
    temp_file = Path("temp_genes.txt")
    with open(temp_file, "w") as f:
        f.write('\n'.join(genes))
        
    cmd = [
        "curl", "-s",
        "-F", f"list=@{temp_file}",
        "-F", f"description={description}",
        "https://maayanlab.cloud/Enrichr/addList"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    user_list_id = json.loads(res.stdout)["userListId"]
    temp_file.unlink()
    
    # 2. Get and save raw JSON responses
    libraries = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
    for lib in libraries:
        params = {
            'userListId': user_list_id,
            'backgroundType': lib
        }
        url = f"https://maayanlab.cloud/Enrichr/enrich?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as response:
            raw_data = response.read().decode('utf-8')
            output_name = f"./aggregated_results/{prefix}_{lib.lower()}_raw.json"
            with open(output_name, "w") as f:
                f.write(raw_data)
            print(f"Saved raw JSON: {output_name}")

# Run queries
save_raw_enrichr(male_genes, "KIRC_Male_Top300", "male_top300")
save_raw_enrichr(female_genes, "KIRC_Female_Top300", "female_top300")
