# engagement_that_sells

Update: 
  - notebook 1: added figure 2 and A.4; moved figures in notebook 4 here
  - notebook 2: added results on incentive alignment
  - notebook 3: added figures 5,6; moved figures in notebook 6 here 
  - notebook 4: deleted, content moved to notebook 1
  - notebook 5: renamed to robustness check
  - notebook 6: deleted, content moved to notebook 3
  
The code contains 4 Jupyter notebooks in R, each performing a main subtask. *To-do: add Python notebooks on model training.* <br/>

Data is stored in tiktok.RData. 

1. Summary statistics

Input data: data used for model construction (model_construction) and evaluation (model_evaluation, search). <br/>
Output data: summary statistics reported in the main text and Appendix F.

<!-- 2. Model training
Input data: data used for model construction (model_construction.csv) and raw video data.
Output data: model performance.  -->

2. Main evaluation results

Input data: data used for model evaluation (model_evaluation, search). <br/>
Output data: evaluation tables and figures in the main text.

3. Drivers of pe-score

Input data: <br/>
Output data:

4. Robustness checks

Input data: data used for model evaluation (model_evaluation, search). <br/>
Output data: tables and figures in Appendix H.

<br/>

Variable dictionary

Computed scores:
  - pe_score: product engagement score
  - p_score: product score
  - e_score: engagement score

Product characteristics:
  - rev: 30-day sales revenue
  - rev_day: imputed daily sales revenue
  - price: the actual price of the product in RMB
  - discount: the amount of discount in RMB, price + discount is the listed/original price
  - category: which category is the product in
  - search: Baidu search index of the product
  - avg_search: Baidu search index of the product averaged over time

Influencer characteristics:
  - gender: influencer gender
  - fans: number of followers
  - avg_play: average number of plays of an influencer's videos
  - influencer_price: the average price charged by an influencer for an ad
  - expected_cpm: expected cost per thousand views
  - order_cnt: number of video ads an influencer has posted
