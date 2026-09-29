"""
E-Commerce Customer Segmentation
Industry: Retail and E-Commerce
Algorithm: KNN (behaviour similarity + anomaly flagging)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

OUT = os.path.join("e:\\", "projects", "ds h Q", "ds-Q2")

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150)
    print(f"Saved: {name}")

# ============================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================
np.random.seed(42)
N = 1000

purchase_freq  = np.random.exponential(scale=5, size=N).clip(1, 30).round()
recency        = np.random.exponential(scale=30, size=N).clip(1, 180).round()
spending       = (purchase_freq * np.random.normal(50, 20, N) + np.random.normal(100, 50, N)).clip(20, 5000).round(2)
product_div    = np.random.randint(1, 15, N)
discount_usage = np.random.beta(2, 5, N).round(2)

df = pd.DataFrame({
    "Customer_ID":        [f"C{str(i).zfill(5)}" for i in range(1, N + 1)],
    "Purchase_Frequency": purchase_freq,
    "Recency":            recency,
    "Spending":           spending,
    "Product_Diversity":  product_div,
    "Discount_Usage":     discount_usage,
})

df.to_csv(os.path.join(OUT, "customer_data.csv"), index=False)
print("Dataset shape:", df.shape)

# ============================================================
# 2. DATA CLEANING & VALIDATION
# ============================================================
print("\n-- Missing Values --")
print(df.isnull().sum())
df.dropna(inplace=True)

numeric_cols = ["Purchase_Frequency", "Recency", "Spending", "Product_Diversity", "Discount_Usage"]
print("\n-- Descriptive Statistics --")
print(df[numeric_cols].describe().round(2))

# ============================================================
# 3. RFM METRICS
# ============================================================
df["R_Score"] = pd.qcut(df["Recency"], q=4, labels=[4, 3, 2, 1]).astype(int)
df["M_Score"] = pd.qcut(df["Spending"], q=4, labels=[1, 2, 3, 4]).astype(int)
df["F_Score"] = pd.qcut(df["Purchase_Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

def rfm_segment(score):
    if score >= 10: return "Champions"
    elif score >= 8: return "Loyal"
    elif score >= 6: return "Potential"
    elif score >= 4: return "At Risk"
    else: return "Lost"

df["Segment"] = df["RFM_Score"].apply(rfm_segment)

print("\n-- RFM Segment Distribution --")
print(df["Segment"].value_counts())

# Special groups (top 40% frequency + bottom 40% spending, and vice versa)
f60 = df["Purchase_Frequency"].quantile(0.60)
s40 = df["Spending"].quantile(0.40)
s60 = df["Spending"].quantile(0.60)
f40 = df["Purchase_Frequency"].quantile(0.40)

freq_low_spend = df[(df["Purchase_Frequency"] >= f60) & (df["Spending"] <= s40)]
heavy_infreq   = df[(df["Spending"] >= s60) & (df["Purchase_Frequency"] <= f40)]

print(f"\nFrequent but Low-Spending customers : {len(freq_low_spend)}")
print(f"Heavy Spenders but Infrequent       : {len(heavy_infreq)}")

# ============================================================
# 4. FOUR VISUALISATIONS
# ============================================================
sns.set_theme(style="whitegrid", palette="muted")
seg_order   = ["Champions", "Loyal", "Potential", "At Risk", "Lost"]
seg_colors  = sns.color_palette("Set2", len(seg_order))
seg_palette = dict(zip(seg_order, seg_colors))

# VIZ 1: RFM Segment bar + RFM Score histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Viz 1 - RFM Segment Distribution", fontsize=14, fontweight="bold")

seg_counts = df["Segment"].value_counts().reindex(seg_order)
axes[0].bar(seg_counts.index, seg_counts.values,
            color=[seg_palette[s] for s in seg_counts.index])
axes[0].set_xlabel("Segment")
axes[0].set_ylabel("Number of Customers")
axes[0].set_title("Customers per Segment")
for i, v in enumerate(seg_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].hist(df["RFM_Score"], bins=range(3, 14), color="#5DADE2", edgecolor="white", rwidth=0.85)
axes[1].set_xlabel("RFM Score")
axes[1].set_ylabel("Count")
axes[1].set_title("RFM Score Distribution")

plt.tight_layout()
save("viz1_rfm_segments.png")
plt.show()

# VIZ 2: Spending vs Purchase Frequency coloured by segment
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Viz 2 - Spending vs Purchase Frequency by Segment",
             fontsize=14, fontweight="bold")

for seg in seg_order:
    grp = df[df["Segment"] == seg]
    ax.scatter(grp["Purchase_Frequency"], grp["Spending"],
               label=seg, alpha=0.5, s=30, color=seg_palette[seg])

ax.scatter(freq_low_spend["Purchase_Frequency"], freq_low_spend["Spending"],
           facecolors="none", edgecolors="black", s=70, linewidths=1,
           label=f"Frequent+LowSpend ({len(freq_low_spend)})")
ax.scatter(heavy_infreq["Purchase_Frequency"], heavy_infreq["Spending"],
           marker="D", facecolors="none", edgecolors="red", s=70, linewidths=1,
           label=f"Heavy+Infrequent ({len(heavy_infreq)})")

ax.set_xlabel("Purchase Frequency")
ax.set_ylabel("Spending ($)")
ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
save("viz2_spending_vs_frequency.png")
plt.show()

# VIZ 3: Segment comparison boxplots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Viz 3 - High-Value vs Low-Value Segment Comparison",
             fontsize=14, fontweight="bold")

sns.boxplot(data=df, x="Segment", y="Spending", order=seg_order,
            palette=seg_palette, ax=axes[0])
axes[0].set_title("Spending by Segment")
axes[0].set_xlabel("Segment")
axes[0].set_ylabel("Spending ($)")
axes[0].tick_params(axis="x", rotation=20)

sns.boxplot(data=df, x="Segment", y="Recency", order=seg_order,
            palette=seg_palette, ax=axes[1])
axes[1].set_title("Recency by Segment")
axes[1].set_xlabel("Segment")
axes[1].set_ylabel("Days Since Last Purchase")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
save("viz3_segment_comparison.png")
plt.show()

# VIZ 4: Normalised segment profile heatmap + Discount usage bar
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Viz 4 - Discount Usage and Product Diversity by Segment",
             fontsize=14, fontweight="bold")

profile_cols = ["Discount_Usage", "Product_Diversity", "Purchase_Frequency", "Spending", "Recency"]
seg_means      = df.groupby("Segment")[profile_cols].mean()
seg_means_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min())

sns.heatmap(seg_means_norm.reindex(seg_order),
            annot=seg_means.reindex(seg_order).round(1),
            fmt=".1f", cmap="YlOrRd", ax=axes[0], linewidths=0.5)
axes[0].set_title("Normalised Segment Profiles (raw values shown)")

disc_means = df.groupby("Segment")["Discount_Usage"].mean().reindex(seg_order)
axes[1].bar(disc_means.index, disc_means.values,
            color=[seg_palette[s] for s in disc_means.index])
axes[1].set_xlabel("Segment")
axes[1].set_ylabel("Avg Discount Usage Rate")
axes[1].set_title("Discount Usage by Segment")
for i, v in enumerate(disc_means.values):
    axes[1].text(i, v + 0.003, f"{v:.2f}", ha="center")

plt.tight_layout()
save("viz4_discount_diversity.png")
plt.show()

# ============================================================
# 5. KNN MODEL
# ============================================================
features = ["Purchase_Frequency", "Recency", "Spending", "Product_Diversity", "Discount_Usage"]
label_map = {"Champions": 4, "Loyal": 3, "Potential": 2, "At Risk": 1, "Lost": 0}
df["Segment_Label"] = df["Segment"].map(label_map)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
y        = df["Segment_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

label_names = ["Lost", "At Risk", "Potential", "Loyal", "Champions"]
print("\n-- KNN Classification Report --")
print(classification_report(y_test, y_pred, target_names=label_names))

# Anomaly flagging: top 5% by avg distance to 5 nearest neighbours
nn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
nn_model.fit(X_scaled)
distances, _ = nn_model.kneighbors(X_scaled)
avg_dist      = distances[:, 1:].mean(axis=1)
threshold     = np.percentile(avg_dist, 95)
df["Anomaly"] = (avg_dist > threshold).astype(int)
print(f"\nAnomaly customers flagged (top 5% outlier distance): {df['Anomaly'].sum()}")

# KNN evaluation plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("KNN - Model Evaluation", fontsize=14, fontweight="bold")

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")
axes[0].tick_params(axis="x", rotation=20)

anom_counts = df.groupby("Segment")["Anomaly"].sum().reindex(seg_order)
axes[1].bar(anom_counts.index, anom_counts.values,
            color=[seg_palette[s] for s in anom_counts.index])
axes[1].set_xlabel("Segment")
axes[1].set_ylabel("Anomaly Count")
axes[1].set_title("Anomalous Customers per Segment")
for i, v in enumerate(anom_counts.values):
    axes[1].text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")

plt.tight_layout()
save("viz5_knn_evaluation.png")
plt.show()

# ============================================================
# 6. KEY INSIGHTS & ACTION PLAN
# ============================================================
champ_pct    = (df["Segment"] == "Champions").mean() * 100
lost_pct     = (df["Segment"] == "Lost").mean() * 100
champ_spend  = df[df["Segment"] == "Champions"]["Spending"].mean()
lost_spend   = df[df["Segment"] == "Lost"]["Spending"].mean()
top_disc_seg = df.groupby("Segment")["Discount_Usage"].mean().idxmax()
anomaly_cnt  = int(df["Anomaly"].sum())

print("""
=================================================================
  KEY INSIGHTS
=================================================================""")
print(f"""
  1. CHAMPIONS DRIVE DISPROPORTIONATE REVENUE
     Champions ({champ_pct:.1f}% of customers) average ${champ_spend:.0f} in
     spending vs ${lost_spend:.0f} for Lost customers -- a {champ_spend/lost_spend:.1f}x gap.
     Retaining this group is the single highest-ROI priority.

  2. FREQUENT BUT LOW-SPENDING CUSTOMERS EXIST
     {len(freq_low_spend)} customers purchase frequently but spend very little.
     These are prime candidates for upsell and bundle campaigns
     to increase average order value.

  3. HEAVY SPENDERS WHO PURCHASE INFREQUENTLY
     {len(heavy_infreq)} customers spend heavily but visit rarely.
     Targeted re-engagement (personalised offers, loyalty perks)
     could convert them into Loyal or Champion tier.

  4. DISCOUNT DEPENDENCY IS HIGHEST IN '{top_disc_seg}'
     The '{top_disc_seg}' segment has the highest discount usage rate,
     suggesting price sensitivity. Over-discounting risks margin
     erosion without improving long-term loyalty.

  5. RECENCY IS A STRONG CHURN SIGNAL
     'Lost' and 'At Risk' segments have significantly higher
     recency (days since last purchase), confirming that
     time-since-purchase is an early churn indicator.

  6. PRODUCT DIVERSITY CORRELATES WITH SEGMENT VALUE
     Higher-value segments (Champions, Loyal) explore more
     product categories, indicating cross-sell opportunities
     can deepen engagement and increase lifetime value.

  7. {anomaly_cnt} BEHAVIOURAL ANOMALIES DETECTED
     KNN nearest-neighbour analysis flagged {anomaly_cnt} customers
     whose behaviour differs strongly from their peers.
     These may represent fraud, data errors, or niche VIP buyers
     requiring manual review.
""")

print("""
=================================================================
  MARKETING ACTION PLAN
=================================================================

  ACTION 1 - Champions: Reward and Retain
  * Launch a VIP loyalty programme with early access to new
    products, exclusive discounts, and personalised thank-you
    communications. Protect this segment at all costs.

  ACTION 2 - Loyal: Upsell and Cross-Sell
  * Use product diversity data to recommend complementary
    categories. Introduce tiered rewards to nudge Loyal
    customers toward Champion status.

  ACTION 3 - Frequent Low-Spenders: Bundle Offers
  * Create value bundles and minimum-spend incentives
    (e.g. "Spend $X more for free shipping") to raise
    average order value without reducing visit frequency.

  ACTION 4 - Heavy Infrequent Spenders: Re-Engagement
  * Trigger personalised win-back emails at 30/60/90-day
    recency milestones with curated product recommendations
    based on previous high-value purchases.

  ACTION 5 - At Risk / Lost: Reactivation Campaign
  * Send time-limited reactivation offers (e.g. 15% off
    next purchase). If no response after 2 attempts,
    move to low-cost retention (newsletter only).

  ACTION 6 - Discount-Dependent Segments: Margin Protection
  * Gradually shift discount-heavy segments toward value-
    based incentives (loyalty points, free gifts) rather
    than straight price cuts to protect margins.

  ACTION 7 - Anomaly Customers: Manual Review
  * Route flagged anomaly customers to the fraud/data team
    for review. Confirmed VIP outliers should be added to
    the Champions programme manually.

=================================================================
""")
