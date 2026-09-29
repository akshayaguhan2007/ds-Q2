# ds-Q2 — E-Commerce Customer Segmentation

## Overview
Retail / E-Commerce data science project that performs behaviour-based customer segmentation using RFM metrics and a KNN model to identify similar behaviour patterns and flag anomalous customers.

## Dataset
1000 synthetic customers with the following features:
- `Purchase_Frequency` — number of purchases made
- `Recency` — days since last purchase
- `Spending` — total spend ($)
- `Product_Diversity` — number of distinct product categories purchased
- `Discount_Usage` — proportion of purchases using a discount

## Project Structure
```
customer_segmentation.py   # Main analysis script
customer_data.csv          # Generated dataset
viz1_rfm_segments.png
viz2_spending_vs_frequency.png
viz3_segment_comparison.png
viz4_discount_diversity.png
viz5_knn_evaluation.png
```

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python customer_segmentation.py
```

## Key Results
| Metric | Value |
|---|---|
| Model | KNN (k=7, Euclidean distance) |
| Overall Accuracy | 70% |
| RFM Segments | 5 (Champions, Loyal, Potential, At Risk, Lost) |
| Champions vs Lost Spend Gap | 5.4x |
| Frequent Low-Spenders | 39 customers |
| Heavy Infrequent Spenders | 7 customers |
| Anomalies Flagged | 50 customers |

## 7 Key Insights
1. Champions (22.2%) average $649 spend vs $121 for Lost — a 5.4x gap
2. 39 customers are frequent but low-spending (upsell targets)
3. 7 customers spend heavily but purchase infrequently (re-engagement targets)
4. Loyal segment has the highest discount dependency — margin risk
5. Recency is a strong churn signal for At Risk and Lost segments
6. Product diversity increases with segment value — cross-sell opportunity
7. 50 behavioural anomalies flagged by KNN for manual review

## Marketing Action Plan
1. Champions: VIP loyalty programme — protect at all costs
2. Loyal: Cross-sell with product diversity data + tiered rewards
3. Frequent Low-Spenders: Bundle offers and minimum-spend incentives
4. Heavy Infrequent: Personalised win-back emails at 30/60/90-day milestones
5. At Risk / Lost: Time-limited reactivation offers
6. Discount-Dependent: Shift to value-based incentives to protect margins
7. Anomalies: Route to fraud/data team for manual review
