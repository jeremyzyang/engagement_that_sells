# engagement_that_sells

The code contains 6 Jupyter notebooks in R, each performing a main subtask. Data is stored in tiktok.RData. *To-do: add Python notebooks on model training.*

1. Summary statistics

Input data: data used for model construction (model_construction) and evaluation (model_evaluation, search). <br/>
Output data: summary statistics reported in the main text.

<!-- 2. Model training
Input data: data used for model construction (model_construction.csv) and raw video data.
Output data: model performance.  -->

2. Main evaluation results

Input data: data used evaluation (model_evaluation, search). <br/>
Output data: evaluation tables and figures in the main text.

3. Drivers of pe-score

*To-do: update the notebook*

4. Appendix F: additional summary statistics

Input data: data used evaluation (model_evaluation). <br/>
Output data: tables and figures in Appendix F.

5. Appendix H: robustness checks

Input data: data used evaluation (model_evaluation, search). <br/>
Output data: tables and figures in Appendix H.

5. Appendix I: additional analysis of drivers

*To-do: update the notebook* 

<br/><br/>

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
