import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path

# Create aggregated_results directory if it doesn't exist
Path("./aggregated_results").mkdir(parents=True, exist_ok=True)

# Load raw rankings
male_df = pd.read_csv("./aggregated_results/male_expr_raw_rra.csv")
female_df = pd.read_csv("./aggregated_results/female_expr_raw_rra.csv")

# Filter out empty or null names
male_genes = male_df["Name"].dropna().head(300).tolist()
female_genes = female_df["Name"].dropna().head(300).tolist()

def get_string_network(genes, filename):
    print(f"Querying STRING API for {len(genes)} genes...")
    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "\n".join(genes),
        "species": 9606,        # Human
        "required_score": 400,  # Medium confidence score >= 0.4
        "caller_identity": "kirc_prognostic_pipeline"
    }
    encoded_data = urllib.parse.urlencode(params).encode('utf-8')
    req = urllib.request.Request(url, data=encoded_data)
    
    try:
        with urllib.request.urlopen(req) as response:
            tsv_data = response.read().decode('utf-8')
            output_path = Path(f"./aggregated_results/{filename}")
            with open(output_path, "w") as f:
                f.write(tsv_data)
            print(f"Saved PPI network to: {output_path}")
            
            # Read first few lines of TSV to verify
            df_test = pd.read_csv(output_path, sep="\t")
            print(f"Found {len(df_test)} protein interactions in the network.")
    except Exception as e:
        print(f"Error querying STRING for {filename}: {e}")

# Run queries
get_string_network(male_genes, "kirc_male_string_ppi.tsv")
get_string_network(female_genes, "kirc_female_string_ppi.tsv")
