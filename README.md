# engagement_that_sells
  
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

Data frames:
  - e_space: engagement scores over space
  - e_time: engagement scores over time
  - p_space: product scores over time
  - p_time: product scores over time
  - pe_space: product engagement scores over space
  - pe_time: product engagement scores over time
  - activity: regression table of engagement on activities
  - emotion: 7 emotions in each second, engagement scores
  - object: number of times an object appeared in high vs. low engagement pixels; count_diff is the difference between the two
  - own: product engagement scores of videos in which influencers advertise for their own product (own = 1) or others' products (own = 0); product characteristics
  - loss: training and validation loss
  - search: Baidu search index of the product (products not listed are not identified on Baidu, typically means search volume is negligible)
  - model_construction: engagement data used for model training, validation, and testing; share_res, like_res, and comment_res are residualized engagement data
  - model_evaluation: sales data used for model evaluation
    - video characteristics:
      - video_full_id: video ids
      - video_posted_time: the date the video is posted
      - len: video length in seconds
      - like, comment, share: number of likes, comments, shares
      - pe_score: product engagement score constructed from engagement heatmaps trained on shares
      - p_score: product score
      - e_score: engagement score
      - pe_score_like: product engagement score constructed from engagement heatmaps trained on likes
      - pe_score_comment: product engagement score constructed from engagement heatmaps trained on comments
      - pe_score_unsup: product engagement scores constructed from unsupervised engagement heatmaps
    -product characteristics:
      - taobao_id: product ids
      - rev: 30-day sales revenue
      - rev_day: imputed daily sales revenue
      - price: the actual price of the product in RMB
      - discount: the amount of discount in RMB, price + discount is the listed/original price
      - category: which category is the product in
      - search: Baidu search index of the product
      - avg_search: Baidu search index of the product averaged over time
    - influencer characteristics:
      - influencer_id: influencer ids  
      - gender: influencer gender
      - fans: number of followers
      - avg_play: average number of plays of an influencer's videos
      - influencer_price: the average price charged by an influencer for an ad
      - expected_cpm: expected cost per thousand views
      - order_cnt: number of video ads an influencer has posted
