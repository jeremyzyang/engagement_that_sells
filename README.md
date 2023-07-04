# engagement_that_sells

The code contains X Jupyter notebooks in R, each performing a main subtask. 

1. Summary statistics

Input data: data used for model construction (model_construction.csv) and evaluation (sales_panel.csv).
Output data: summary statistics reported in the paper.

2. Model training

Input data: data used for model construction (model_construction.csv) and raw video data.
Output data: model performance. 

4. Main evaluation results

Input data: data used evaluation (sales_panel.csv).
Output data: tables and figures in the main text.

4. Robustness checks

Input data: data used evaluation (sales_panel.csv).
Output data: tables and figures in the appendices.\\


Variable dictionary

Computed scores:
  - pe_score: product engagement score
  - p_score: product score
  - e_score: engagement score

Product characteristics:
  - price: the actual price of the product in RMB
  - discount: the amount of discount in RMB, price + discount is the listed/original price
  - category: which category is the product in
  - search: Baidu search index of the product

Influencer characteristics:
  - 
