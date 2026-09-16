library(RobustRankAggreg)
library(dplyr)

folder <- "C:\\Users\\joepi\\Downloads\\results_folded_2"
files <- list.files(folder, pattern="\\.csv$", full.names=TRUE)

# function to extract rankings from files
extract_rankings <- function(files_subset) {
  
  all_rank_lists <- list()
  
  for (file in files_subset) {
    
    df <- read.csv(file, stringsAsFactors = FALSE)
    
    # split by voter
    split_lists <- split(df, df$Voter)
    
    # convert each voter to ranked vector
    rank_lists <- lapply(split_lists, function(x) {
      x <- x[order(x$Rank), ]
      x$Item
    })
    
    # make voter names unique across files
    names(rank_lists) <- paste(basename(file), names(rank_lists), sep="_")
    
    all_rank_lists <- c(all_rank_lists, rank_lists)
  }
  
  return(all_rank_lists)
}

# separate files by gender
male_files   <- files[grepl("male", basename(files), ignore.case=TRUE)]
female_files <- files[grepl("female", basename(files), ignore.case=TRUE)]

# extract rankings
male_rank_lists   <- extract_rankings(male_files)
female_rank_lists <- extract_rankings(female_files)

# run RRA
male_rra   <- aggregateRanks(male_rank_lists)
female_rra <- aggregateRanks(female_rank_lists)

# extract significant gene names
male_sig_genes   <- male_rra$Name[male_rra$Score < 0.05]
female_sig_genes <- female_rra$Name[female_rra$Score < 0.05]
cat(length(male_sig_genes))
cat(length(female_sig_genes))

# compute overlap
overlap_genes <- intersect(male_sig_genes, female_sig_genes)

# count overlap
overlap_count <- length(overlap_genes)

# print overlap count
cat("Overlapping significant genes:", overlap_count, "\n")

male_only <- setdiff(male_sig_genes, female_sig_genes)
female_only <- setdiff(female_sig_genes, male_sig_genes)

male_rra$ConsensusRank   <- seq_len(nrow(male_rra))
female_rra$ConsensusRank <- seq_len(nrow(female_rra))

male_only_df <- male_rra[male_rra$Name %in% male_only, 
                         c("Name", "ConsensusRank", "Score")]

female_only_df <- female_rra[female_rra$Name %in% female_only, 
                             c("Name", "ConsensusRank", "Score")]

write.csv(male_only_df, file = "C:\\Users\\joepi\\Code\\Thesis\\spring_2026\\aggregated_results\\male_expr_aggregated_ranking.csv", row.names = FALSE)

write.csv(female_only_df, file = "C:\\Users\\joepi\\Code\\Thesis\\spring_2026\\aggregated_results\\female_expr_aggregated_ranking.csv", row.names = FALSE)

print(female_only_df$Name)
print(male_only_df)
