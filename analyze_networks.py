import pandas as pd

def analyze_ppi_network(tsv_path, cohort_name):
    print(f"\n=========================================")
    print(f"=== PPI NETWORK ANALYSIS: {cohort_name} ===")
    print(f"=========================================")
    
    # Load network TSV
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        print(f"Error loading {tsv_path}: {e}")
        return
        
    if df.empty:
        print("No interactions found in the network.")
        return
        
    # Count degree centrality (occurrences in preferredName_A and preferredName_B)
    all_nodes = pd.concat([df["preferredName_A"], df["preferredName_B"]])
    degree_counts = all_nodes.value_counts()
    
    print(f"\nTotal interacting proteins: {len(degree_counts)}")
    print(f"Total functional interactions (edges): {len(df)}")
    
    print("\nTop 10 Hub Genes (Highest Degree Centrality):")
    for rank, (gene, degree) in enumerate(degree_counts.head(10).items(), 1):
        print(f" {rank}. {gene} ({degree} connections)")
        
    # Find highly interconnected sub-networks (interconnected components)
    # A simple way to find MCODE-like tight modules is to look at genes with highest degrees and their connections
    print(f"\nTop Interconnected Modules:")
    top_hubs = degree_counts.head(5).index.tolist()
    for hub in top_hubs:
        connections = df[(df["preferredName_A"] == hub) | (df["preferredName_B"] == hub)]
        neighbors = set(connections["preferredName_A"]).union(set(connections["preferredName_B"])) - {hub}
        neighbors_list = list(neighbors)[:8] # Show up to 8 neighbors
        print(f" * Hub [{hub}] coordinates a module with: {', '.join(neighbors_list)}")

# Run analysis
analyze_ppi_network("./aggregated_results/kirc_male_string_ppi.tsv", "KIRC Male (Top 300)")
analyze_ppi_network("./aggregated_results/kirc_female_string_ppi.tsv", "KIRC Female (Top 300)")
