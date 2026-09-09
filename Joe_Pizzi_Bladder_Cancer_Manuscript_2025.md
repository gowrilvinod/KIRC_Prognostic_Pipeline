# Machine-learning-based determination of sex-related bladder cancer biomarkers

**Authors:** Joseph R. Pizzi¹, Image Adhikari², Prakyat Prakash², Hiroshi Miyamoto³*, Feng Cui¹*  
**Affiliations:**  
¹ Thomas H. Gosnell School of Life Sciences, College of Science, Rochester Institute of Technology, Rochester, NY, USA  
² Department of Computer Science, Golisano College of Computing and Information Sciences, Rochester Institute of Technology, Rochester, NY, USA  
³ Departments of Pathology & Laboratory Medicine and Urology, University of Rochester Medical Center, Rochester, NY 14642, USA  

**Corresponding Authors:**  
* Feng Cui: fxcsbi@rit.edu  
* Hiroshi Miyamoto: Hiroshi_Miyamoto@urmc.rochester.edu  

**bioRxiv Preprint DOI:** https://doi.org/10.64898/2025.12.20.695175  
**License:** CC-BY-NC-ND 4.0 International  
**Keywords:** Bladder cancer; machine learning; feature selection; gene expression; sex hormones; sex dimorphism  

---

## Abstract

Bladder cancer exhibits sex-specific behavior, occurring more frequently in males but progressing to advanced stages more commonly in females. The activation of sex hormone receptors may explain these differences, but the exact genetic drivers remain poorly understood. Furthermore, current bladder cancer biomarkers have inconsistent sensitivities and specificities in practice, making early diagnosis a challenge. 

This study approaches bladder cancer biomarker discovery through machine learning techniques on gender and disease-stratified RNA-seq data. Training sets limited to differentially expressed genes were subjected to four different feature selection methods: differential gene expression analysis adjusted p-value, recursive feature elimination with support vector machine (SVM-RFE), logistic regression, and an optimized random forest (RF) procedure. Gene panels were compared and aggregated across selection strategies and cross-validation folds using Robust Rank Aggregation (RRA) to identify robust biomarkers for sex-specific bladder cancer development and progression. 

When applied to unseen external datasets (GSE236932 and GSE188715) and limited to 50 genes or less, male and female-specific panels achieved areas under the receiver operating characteristic curve (AUROC) of 0.932 and 0.914, respectively, in distinguishing bladder cancer samples from non-tumor controls. Genes such as PRAC1 and PCDH11Y were identified as high-impact predictors related to sex hormones or chromosomes for male tumor development. In the female-specific panel, genes related to aberrant androgen signaling across tumor types like AR, PLXNA1, USP54, and PMEPA1 were influential. These results offer potential targets for further in vivo/vitro experimentation and provide a framework for constructing generalizable, high-performance gene panels for bladder cancer diagnosis and prognosis.

---

## Introduction

Bladder cancer (BCa) is the 6th most common cancer in the United States, representing 4.2% of all cancer cases. The 5-year relative survival rate is estimated at 79.0%, but individual prognosis depends heavily on the stage and aggressiveness of the tumor. However, identifying BCa early in development can be difficult, with most symptomatic individuals presenting with gross hematuria, a malady common in other diseases like urinary tract and kidney infections. The current diagnostic standard is cystoscopy, yet this method relies on human judgment and can miss carcinoma in situ without further imaging. Cystoscopy is often preceded by urine cytology to identify abnormal cells characteristic of high-grade tumors, but this technique can prove ineffective for lower-grade cases. FDA-approved molecular biomarkers like NMP22 and BTA exhibit highly variable sensitivities and specificities with high false-positive rates.

Gender-specific differences in frequency and pathogenesis add another layer of complexity. Incidence rate in males is approximately 3.3 times that observed in females. Despite the lower frequency, females tend to have more aggressive forms of BCa upon diagnosis. These findings suggest molecular mechanisms drive gender-specific differences in BCa outcomes. Androgen receptor (AR) and estrogen receptor (ER) signaling, alongside X/Y chromosome escape genes (e.g. KDM6A), play crucial roles in BCa sex dimorphism.

To discover robust, sensitive sex-specific BCa biomarkers, this study integrated gender- and disease-stratified RNA-seq datasets (TCGA-BLCA, GTEx, GSE133624), applied DGEA with DESeq2, four machine learning feature selection methods (Optimized RF, SVM-RFE, Logistic Regression, DGEA adj p-value), aggregated rankings with RRA across 5-fold cross-validation, and validated top gene panels on independent external cohorts (GSE236932 and GSE188715).

---

## Key Results Summary

### 1. Dataset Merging & Normalization
* Merged TCGA-BLCA, GTEx, and GSE133624 into a balanced training set using independent z-score normalization per dataset before concatenation to eliminate batch effects (demonstrated via UMAP).

### 2. Differentially Expressed Genes (DEGs) & Feature Selection
* **Male-specific tumor development:** 2,800 DEGs (1,414 upregulated, 1,386 downregulated).
* **Female-specific tumor development:** 821 DEGs (525 upregulated, 296 downregulated).
* **Sex-related BCa progression (Male vs. Female tumors):** 328 DEGs (223 upregulated, 105 downregulated).
* Aggregated top 300 gene panels constructed using Robust Rank Aggregation (RRA) across 20 input rankings (4 selection methods x 5 folds).

### 3. External Validation Performance (Top ≤50 Gene Panels)
| Task | Top N Genes | Best Model | Balanced Accuracy | F1 Score | AUROC | Composite Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Male Tumor vs. Non-Tumor** | Top 50 | Random Forest | 0.896 | **0.917** | **0.923** | **0.915** |
| **Female Tumor vs. Non-Tumor** | Top 10 | Random Forest | 0.750 | **0.878** | **0.914** | **0.863** |
| **Male Tumor vs. Female Tumor** | Top 25 | SVM | 0.689 | 0.853 | 0.734 | 0.784 |

### 4. Key Biomarkers & Biological Insights
* **Male Development Biomarkers:** `PRAC1` (androgen co-regulator), `PCDH11Y` (Y-linked gene), `TTTY10` (Y-linked lncRNA), `IL1RAPL1` (X-linked gene), `NDNF`, `TOX2`, `MIR222`, `ESPL1` (mitotic anaphase hub gene).
* **Female Development Biomarkers:** `AR` (Androgen Receptor), `PLXNA1`, `USP54`, `PMEPA1`, `ARSF` (X-linked gene), `COL5A2` (ECM collagen hub gene).
* **Male vs. Female Tumor Biomarkers:** `SLC7A11`, `SLC2A14`, `CYP1A2`, `UGT2B15` (xenobiotic and steroid hormone metabolism pathways).

---

## Code Availability

The code used to generate the results in this study is available on GitHub:  
https://github.com/rit-cui-lab/Machine-learning-based-determination-of-sex-related-bladder-cancer-biomarkers

---

## Funding & Acknowledgements

This research was supported by NIH federal grants **R15GM149587** and **R21GM152740**.
