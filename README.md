# **A statistical framework for evaluating repeatability and reproducibility of large language models in diagnostic reasoning**

This repository contains the following code and data:

1. **Calculate_Scores_Per_Run.py**: This file computes the embeddings and average entropy per run. The outputs are .pkl files containing the results for each run. Please note that "Diagnostic reasoning prompts reveal the potential for large language model interpretability in medicine Supplement 1.csv" is a publicly available supplementary file published by Savage, Thomas, et al. "Diagnostic reasoning prompts reveal the potential for large language model interpretability in medicine." NPJ Digital Medicine 7.1 (2024): 20. https://doi.org/10.1038/s41746-024-01010-1.
3. **Calculate_Repeatability_Scores.py**: This is a post-processing file that takes the .pkl files generated using Calculate_Scores_Per_Run.py and calculates the repeatatbility scores.
4. **Calculate_Reproducibility_Scores.py**: This is a post-processing file that takes the .pkl files generated using Calculate_Scores_Per_Run.py and calculates the reproducibility scores.

